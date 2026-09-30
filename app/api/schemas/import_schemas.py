"""
导入接口 Pydantic 模型：/import 与 /import/progress 共用的响应结构。
"""
from pydantic import BaseModel, Field


class ImportResponse(BaseModel):
    """导入任务受理响应：返回任务ID供进度查询/SSE订阅。"""
    task_id: str = Field(..., description="导入任务ID")


class ProgressResponse(BaseModel):
    """导入进度响应：状态 + 已完成节点（中文）+ 运行中节点（中文）。"""
    status: str = Field(..., description="任务状态：pending/processing/completed/failed")
    done_list: list[str] = Field(default_factory=list, description="已完成节点（中文展示名）")
    running_list: list[str] = Field(default_factory=list, description="运行中节点（中文展示名）")


# ---------------- 切片管理（/chunks） ----------------

class ChunkItem(BaseModel):
    """单个切片实体（含 Milvus 主键）。"""
    id: int | None = Field(default=None, description="Milvus 主键 id（插入结果不回传时暂缺）")
    chunk_text: str = Field(..., description="切片正文")
    file_title: str = Field(..., description="所属文档标题")
    parent_title: str = Field(..., description="父级标题（章节）")
    part: int = Field(..., description="切片序号")


class ChunkDocument(BaseModel):
    """按文档聚合的切片统计。"""
    file_title: str = Field(..., description="文档标题")
    chunk_count: int = Field(..., description="切片数量")


class ChunkDocumentListResponse(BaseModel):
    """文档列表响应。"""
    documents: list[ChunkDocument] = Field(default_factory=list, description="文档统计列表")


class ChunkListResponse(BaseModel):
    """切片分页列表响应。"""
    total: int = Field(..., description="该文档切片总数")
    page: int = Field(..., description="当前页码（从1起）")
    page_size: int = Field(..., description="每页条数")
    items: list[ChunkItem] = Field(default_factory=list, description="切片列表")


class ChunkUpdateRequest(BaseModel):
    """编辑切片请求：正文必填，标题/序号可选（不传沿用旧值）。"""
    chunk_text: str = Field(..., min_length=1, max_length=8192, description="新的切片正文")
    parent_title: str | None = Field(default=None, max_length=512, description="新的父级标题（不传沿用旧值）")
    part: int | None = Field(default=None, description="新的切片序号（不传沿用旧值）")


class ChunkCreateRequest(BaseModel):
    """手动新增切片请求。"""
    file_title: str = Field(..., min_length=1, max_length=512, description="所属文档标题")
    parent_title: str = Field(default="", max_length=512, description="父级标题")
    part: int = Field(default=1, description="切片序号")
    chunk_text: str = Field(..., min_length=1, max_length=8192, description="切片正文")


class ChunkMutationResponse(BaseModel):
    """编辑/新增切片响应：编辑为删旧插新，新旧 id 都返回。"""
    old_id: int | None = Field(default=None, description="旧切片 id（仅编辑时有值）")
    new_id: int | None = Field(default=None, description="新切片 id（Milvus auto_id）")
    chunk: ChunkItem = Field(..., description="写库后的切片实体")


if __name__ == '__main__':
    # 模型自检：合法构造 + 默认值
    imp = ImportResponse(task_id="test_task_001")
    prog = ProgressResponse(status="processing", done_list=["检查文件"], running_list=["PDF转Markdown"])
    print(f"导入响应OK: {imp.model_dump_json()}")
    print(f"进度响应OK: {prog.model_dump_json()}")
    prog_default = ProgressResponse(status="pending")
    print(f"默认进度OK: done={len(prog_default.done_list)}, running={len(prog_default.running_list)}")
