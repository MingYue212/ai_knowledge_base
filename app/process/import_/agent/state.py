"""
导入 Agent 的 LangGraph 状态定义与工厂函数。
ImportGraphState 为 TypedDict，贯穿整个导入流水线各节点。
"""
import copy
from typing import TypedDict
import json


class ImportGraphState(TypedDict):
    """导入流程状态：文件路径、Markdown 内容、切片结果、向量化上下文等。"""
    task_id: str

    local_file_path: str

    md_path: str
    pdf_path: str
    local_dir: str
    file_title: str

    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    md_content: str

    chunks: list[dict]
    item_name: str
    embedding_context: list[dict]


# 模板对象
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embedding_context": [],
}


def create_default_state(**args) -> ImportGraphState:
    """深拷贝默认状态模板并用传入参数覆盖，返回新状态。"""
    deep_new_state = copy.deepcopy(graph_default_state)
    deep_new_state.update(args)
    return deep_new_state


def get_default_state() -> ImportGraphState:
    """获取只读的默认状态模板。"""
    return graph_default_state


if __name__ == '__main__':
    state = create_default_state(task_id="007", local_file_path="./烫金机.pdf")

    print(f"{json.dumps(state, indent=4, ensure_ascii=False)}")
