"""
切片管理服务：已入库 chunk 的查看 / 编辑 / 删除 / 新增（API 层 /chunks 端点的业务实现）。

核心设计（与导入链同源，保证向量一致性）：
- 重嵌复用 embedding_utils.generate_embeddings（文档侧 encode_documents，
  模型原生 L2 归一化）——编辑/新增后的向量与批量导入的向量分布完全一致
- 集合为 auto_id=True，Milvus 不允许带主键 upsert——"编辑"的实现是
  重嵌新文本 → 插入新实体 → 删除旧实体（id 会变化）
- 顺序防御：先插新再删旧。若删除失败只留下重复切片（可在本功能里再删），
  不会出现"编辑失败反而丢数据"
- 过滤表达式用 escape_milvus_string 转义文档标题，防注入/解析错误
- item_name 主体索引不受本服务影响（编辑/删除切片不动主体名向量）
"""
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.shared.model.embedding_utils import generate_embeddings
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string

# 与 chunks 集合 schema 的 VARCHAR 上限对齐（编辑/新增入参校验）
CHUNK_TEXT_MAX = 8192
TITLE_MAX = 512
# Milvus query 单次扫描上限（演示规模的知识库足够一次取全量后客户端聚合/排序）
_QUERY_SCAN_LIMIT = 16384


class ChunkNotFoundError(ValueError):
    """按 id 找不到切片（API 层转 404）。"""


def _build_entity(file_title: str, parent_title: str, part: int | None, chunk_text: str) -> dict:
    """校验入参（对齐 schema 上限）并生成含双向量的入库实体；校验失败抛 ValueError。"""
    file_title = (file_title or "").strip()
    parent_title = (parent_title or "").strip()
    chunk_text = (chunk_text or "").strip()
    if not chunk_text:
        raise ValueError("chunk_text不能为空!")
    if len(chunk_text) > CHUNK_TEXT_MAX:
        raise ValueError(f"chunk_text超长:{len(chunk_text)}>{CHUNK_TEXT_MAX}!")
    if len(file_title) > TITLE_MAX:
        raise ValueError(f"file_title超长:{len(file_title)}>{TITLE_MAX}!")
    if len(parent_title) > TITLE_MAX:
        raise ValueError(f"parent_title超长:{len(parent_title)}>{TITLE_MAX}!")

    embeddings = generate_embeddings([chunk_text])
    return {
        "chunk_text": chunk_text,
        "file_title": file_title or "未命名文档",
        "parent_title": parent_title or "未命名",
        "part": int(part or 1),
        "sparse_vector": embeddings["sparse"][0],
        "dense_vector": embeddings["dense"][0],
    }


@step_log("list_chunk_documents")
def list_chunk_documents() -> list[dict]:
    """统计各文档的切片数量（file_title 去重聚合，按标题排序）。"""
    rows = milvus_gateway.query_chunks(filter_expr="", limit=_QUERY_SCAN_LIMIT)
    counter: dict[str, int] = {}
    for row in rows:
        title = str(row.get("file_title") or "")
        counter[title] = counter.get(title, 0) + 1
    docs = [{"file_title": title, "chunk_count": count} for title, count in counter.items()]
    docs.sort(key=lambda doc: doc["file_title"])
    logger.info(f"切片文档统计完成,共{len(docs)}个文档,{len(rows)}个切片")
    return docs


@step_log("list_chunks")
def list_chunks(file_title: str, page: int = 1, page_size: int = 20) -> dict:
    """按文档标题列出切片，按 (parent_title, part, id) 排序后分页。

    防御性校验：file_title 为空无法定位文档，直接抛错（API 层转 400）。
    """
    title = (file_title or "").strip()
    if not title:
        raise ValueError("file_title不能为空,无法定位文档切片!")
    expr = f'file_title == "{escape_milvus_string(title)}"'
    rows = milvus_gateway.query_chunks(filter_expr=expr, limit=_QUERY_SCAN_LIMIT)

    # Milvus query 无稳定排序，客户端排序后再分页（保证翻页结果稳定）
    rows.sort(key=lambda row: (
        str(row.get("parent_title") or ""),
        int(row.get("part") or 0),
        int(row.get("id") or 0),
    ))
    total = len(rows)
    page = max(1, int(page))
    page_size = min(max(1, int(page_size)), 100)
    start = (page - 1) * page_size
    items = rows[start:start + page_size]
    logger.info(f"切片列表完成,文档:{title},总数:{total},返回第{page}页{len(items)}条")
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@step_log("update_chunk")
def update_chunk(
    chunk_id: int,
    chunk_text: str,
    parent_title: str | None = None,
    part: int | None = None,
) -> dict:
    """编辑切片正文/标题：重嵌 → 插入新实体 → 删除旧实体，返回 {old_id, new_id, chunk}。

    防御性校验：
    1. 切片不存在 —— 抛 ChunkNotFoundError（API 层转 404）
    2. 先插新再删旧 —— 删除失败仅告警（留重复切片可再删，不丢数据）
    """
    old = milvus_gateway.get_chunk(int(chunk_id))
    if old is None:
        raise ChunkNotFoundError(f"切片不存在:id={chunk_id}")

    new_parent = (parent_title if parent_title is not None else str(old.get("parent_title") or "")).strip()
    new_part = int(part) if part is not None else int(old.get("part") or 1)
    entity = _build_entity(str(old.get("file_title") or ""), new_parent, new_part, chunk_text)

    new_id = milvus_gateway.insert_chunk(entity)
    try:
        milvus_gateway.delete_chunks([int(chunk_id)])
    except Exception as e:
        # 旧切片删除失败：保留现场并高亮告警（重复切片可在管理页再删），不抛出
        logger.error(f"旧切片删除失败,存在重复切片!old_id:{chunk_id},new_id:{new_id}:{str(e)}")
    logger.info(f"切片编辑完成,old_id:{chunk_id} -> new_id:{new_id}")
    return {"old_id": int(chunk_id), "new_id": new_id, "chunk": {"id": new_id, **entity}}


@step_log("delete_chunk")
def delete_chunk(chunk_id: int) -> None:
    """按 id 删除单个切片；不存在时抛 ChunkNotFoundError（API 层转 404）。"""
    if milvus_gateway.get_chunk(int(chunk_id)) is None:
        raise ChunkNotFoundError(f"切片不存在:id={chunk_id}")
    milvus_gateway.delete_chunks([int(chunk_id)])
    logger.info(f"切片删除完成,id:{chunk_id}")


@step_log("add_chunk")
def add_chunk(file_title: str, parent_title: str, part: int | None, chunk_text: str) -> dict:
    """手动新增切片：校验 + 重嵌 + 插入，返回带新 id 的切片实体。"""
    entity = _build_entity(file_title, parent_title, part, chunk_text)
    new_id = milvus_gateway.insert_chunk(entity)
    logger.info(f"切片新增完成,new_id:{new_id},文档:{entity['file_title']}")
    return {"id": new_id, **entity}


if __name__ == '__main__':
    # 离线自测：假网关 + 假向量（不需要 Milvus / BGE-M3），验证增删改查与防御逻辑
    import app.rag.import_.chunk_manage_service as svc

    class _FakeGateway:
        def __init__(self):
            self.rows: dict[int, dict] = {}
            self.next_id = 100

        def query_chunks(self, filter_expr, limit, offset=0):
            return [dict(row) for row in self.rows.values()]

        def get_chunk(self, chunk_id):
            return dict(self.rows[chunk_id]) if chunk_id in self.rows else None

        def delete_chunks(self, ids):
            for i in ids:
                self.rows.pop(i, None)

        def insert_chunk(self, entity):
            new_id = self.next_id
            self.next_id += 1
            self.rows[new_id] = {"id": new_id, **entity}
            return new_id

    svc.milvus_gateway = _FakeGateway()
    svc.generate_embeddings = lambda texts: {
        "dense": [[0.1] * 4 for _ in texts],
        "sparse": [{1: 0.5} for _ in texts],
    }

    # 新增 → 列表
    added = svc.add_chunk("烫金机手册", "使用步骤", 2, "烫金机使用前请预热5分钟。")
    assert added["id"] == 100 and added["file_title"] == "烫金机手册"
    docs = svc.list_chunk_documents()
    assert docs == [{"file_title": "烫金机手册", "chunk_count": 1}], docs
    listed = svc.list_chunks("烫金机手册")
    assert listed["total"] == 1 and listed["items"][0]["chunk_text"].startswith("烫金机")

    # 编辑 → 删旧插新（id 变化），parent_title/part 未传时沿用旧值
    updated = svc.update_chunk(100, "预热后再放入物料，温度200度。")
    assert updated["old_id"] == 100 and updated["new_id"] == 101
    assert updated["chunk"]["parent_title"] == "使用步骤" and updated["chunk"]["part"] == 2
    assert svc.milvus_gateway.get_chunk(100) is None

    # 不存在 → ChunkNotFoundError
    try:
        svc.update_chunk(999, "任意")
        raise AssertionError("应抛 ChunkNotFoundError")
    except svc.ChunkNotFoundError:
        pass

    # 删除 → 超长校验
    svc.delete_chunk(101)
    assert not svc.milvus_gateway.rows
    try:
        svc.add_chunk("手册", "章节", None, "超长" * 5000)
        raise AssertionError("应抛超长 ValueError")
    except ValueError as e:
        assert "超长" in str(e)

    print("CHUNK_MANAGE_SERVICE_SELFTEST_ALL_PASSED")
