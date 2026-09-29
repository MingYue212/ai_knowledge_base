"""
检索节点：查询向量化 + Milvus 混合检索。

节点链位置：node_item_name_confirm → [node_search_embedding] → node_rerank
职责：调用 search_service 用改写后的问题做稠密+稀疏混合检索，
召回切片写入 state["search_context"]。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query.search_service import search_chunks
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 切片搜索 (node_search_embedding)
    为什么叫这个名字: 把改写后的问题编码成向量，到 Milvus 做混合检索召回切片。
    """
    add_running_task(state["task_id"], "node_search_embedding")
    state = search_chunks(state)
    add_done_task(state["task_id"], "node_search_embedding")
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始node_search_embedding节点单元测试 =====")
    # 注意：本测试需要 Milvus 已启动（MILVUS_URL）且 BGE-M3 模型可用
    test_state = create_default_state(task_id="test_search_001", query="烫金机怎么使用？")
    result_state = node_search_embedding(test_state)
    search_context = result_state.get("search_context") or []
    logger.info(f"测试完成,召回切片数:{len(search_context)}")
    for idx, hit in enumerate(search_context[:3], start=1):
        logger.info(f"Top{idx}: 来源={hit.get('file_title')} > {hit.get('parent_title')}")
