"""
查询流程 LangGraph 状态定义与工厂函数。
QueryGraphState 为 TypedDict，贯穿整个查询流水线各节点。
"""
import copy
from typing import TypedDict


class QueryGraphState(TypedDict):
    """查询流程状态：原始问题、历史对话、改写结果、检索与精排上下文、最终答案。"""
    task_id: str

    query: str
    history: list[dict]

    item_names: list[str]
    rewritten_query: str

    search_context: list[dict]
    hyde_answer: str
    hyde_context: list[dict]
    rerank_context: list[dict]

    answer: str


# 模板对象
query_default_state: QueryGraphState = {
    "task_id": "",
    "query": "",
    "history": [],
    "item_names": [],
    "rewritten_query": "",
    "search_context": [],
    "hyde_answer": "",
    "hyde_context": [],
    "rerank_context": [],
    "answer": "",
}


def create_default_state(**args) -> QueryGraphState:
    """深拷贝默认状态模板并用传入参数覆盖，返回新状态。"""
    deep_new_state = copy.deepcopy(query_default_state)
    deep_new_state.update(args)
    return deep_new_state


def get_default_state() -> QueryGraphState:
    """获取只读的默认状态模板。"""
    return query_default_state


if __name__ == '__main__':
    state = create_default_state(task_id="task_001", query="烫金机怎么使用？")

    print(f"{state}")
