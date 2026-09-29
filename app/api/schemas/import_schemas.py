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


if __name__ == '__main__':
    # 模型自检：合法构造 + 默认值
    imp = ImportResponse(task_id="test_task_001")
    prog = ProgressResponse(status="processing", done_list=["检查文件"], running_list=["PDF转Markdown"])
    print(f"导入响应OK: {imp.model_dump_json()}")
    print(f"进度响应OK: {prog.model_dump_json()}")
    prog_default = ProgressResponse(status="pending")
    print(f"默认进度OK: done={len(prog_default.done_list)}, running={len(prog_default.running_list)}")
