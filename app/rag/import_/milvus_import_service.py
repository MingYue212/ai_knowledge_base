"""
文档切块 Milvus 入库服务：建集合（幂等）+ 混合向量批量插入。

职责：
1. 组装实体：chunk 业务字段 + dense/sparse 混合向量
2. 确保集合存在（不存在则按 schema 创建并建索引）
3. 批量插入 chunks 集合；若 state["item_name"] 非空，
   额外把主体名称向量化后写入主体名称集合（主体索引）
"""
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.model.embedding_utils import generate_embeddings
from app.shared.runtime.logger import logger, step_log


@step_log("build_chunk_entities")
def build_chunk_entities(state: ImportGraphState) -> list[dict]:
    """把 chunks 与 embedding_context 按索引对齐，组装成 Milvus 入库实体。

    防御性校验：
    1. chunks 与 embedding_context 数量一致 —— 不一致说明向量化环节异常，
       此时入库会错位，必须提前终止
    2. part 统一转 int —— split_service 产出的 part 为整型，这里兜底强转
    """
    # 1.获取参数
    chunks: list[dict] = state.get("chunks") or []
    embedding_context: list[dict] = state.get("embedding_context") or []

    # 2.数量一致性校验
    if len(chunks) != len(embedding_context):
        logger.error(
            f"chunks与embedding_context数量不一致!chunks={len(chunks)},"
            f"embedding_context={len(embedding_context)}"
        )
        raise ValueError("chunks与embedding_context数量不一致!无法组装入库实体,提前终止!!")

    # 3.逐条组装实体，字段名与milvus_gateway的chunks集合schema一一对应
    entities: list[dict] = []
    for chunk, embedding in zip(chunks, embedding_context):
        entities.append({
            "chunk_text": chunk.get("content", ""),
            "file_title": chunk.get("file_title", ""),
            "parent_title": chunk.get("parent_title", ""),
            "part": int(chunk.get("part", 1)),
            "sparse_vector": embedding.get("sparse") or {},
            "dense_vector": embedding.get("dense") or [],
        })
    logger.info(f"完成{len(entities)}个入库实体的组装")
    return entities


@step_log("import_chunks_to_milvus")
def import_chunks_to_milvus(state: ImportGraphState) -> ImportGraphState:
    """切块向量入库 Milvus，并按需建立主体名称索引。

    流程：
    1. 组装入库实体（chunks + 混合向量）
    2. 确保 chunks 集合存在后批量插入
    3. item_name 非空时：向量化名称 → 确保主体名称集合存在 → 插入
    """
    # 1.组装实体
    entities: list[dict] = build_chunk_entities(state)
    if not entities:
        logger.warning("没有可入库的实体,跳过Milvus导入!")
        return state

    # 2.确保集合存在并批量插入（稠密维度取自实际向量，不写死）
    dense_dim = len(entities[0]["dense_vector"])
    milvus_gateway.ensure_chunks_collection(dense_dim)
    milvus_gateway.insert_chunks(entities)
    logger.info(f"完成{len(entities)}个切块的Milvus入库!")

    # 3.主体名称索引：item_name为空时跳过（第0步识别节点为直通实现，通常为空）
    item_name: str = (state.get("item_name") or "").strip()
    if item_name:
        name_embeddings = generate_embeddings([item_name])
        name_entity = {
            "item_name": item_name,
            "dense_vector": name_embeddings["dense"][0],
        }
        milvus_gateway.ensure_item_name_collection(len(name_entity["dense_vector"]))
        milvus_gateway.insert_item_name(name_entity)
        logger.info(f"完成主体名称[{item_name}]的向量索引!")
    else:
        logger.info("item_name为空,跳过主体名称索引!")
    return state
