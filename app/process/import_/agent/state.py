import copy
from typing import TypedDict
import json


class ImportGraphState(TypedDict):
    task_id: str

    local_file_path: str

    md_path: str
    pdf_path: str
    local_dir: str
    file_title: str

    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    md_context: str

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
    "md_context": "",
    "chunks": [],
    "item_name": "",
    "embedding_context": [],
}


# 方法1:根据传入的参数创建一个对应的state
def create_default_state(**args) -> ImportGraphState:
    deep_new_state = copy.deepcopy(graph_default_state)
    deep_new_state.update(args)
    return deep_new_state


# 方法2:获取创建好的默认的空state
def get_default_state() -> ImportGraphState:
    return graph_default_state


if __name__ == '__main__':
    state = create_default_state(task_id="007", local_file_path="./烫金机.pdf")

    print(f"{json.dumps(state, indent=4, ensure_ascii=False)}")
