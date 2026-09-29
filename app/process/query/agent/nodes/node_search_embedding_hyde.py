"""
HyDE 检索节点：假设性文档增强检索（主路检索的"补路"）。

节点链位置：node_search_embedding → [node_search_embedding_hyde] → node_rrf
职责：调用 hyde_service 生成假设性回答并以文搜文，
召回切片写入 state["hyde_context"]，供 node_rrf 与主路结果做排名融合。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query.hyde_service import search_by_hyde
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 切片搜索(假设性文档) (node_search_embedding_hyde)
    为什么叫这个名字: 先让 LLM 生成假设性回答，再用这段回答去 Milvus 检索切片。
    """
    add_running_task(state["task_id"], "node_search_embedding_hyde")
    state = search_by_hyde(state)
    add_done_task(state["task_id"], "node_search_embedding_hyde")
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始node_search_embedding_hyde节点单元测试 =====")
    # 注意：本测试需要 LLM API、Milvus（MILVUS_URL）且 BGE-M3 模型可用
    test_state = create_default_state(task_id="test_hyde_node_001", query="烫金机怎么使用？")
    result_state = node_search_embedding_hyde(test_state)
    hyde_context = result_state.get("hyde_context") or []
    logger.info(f"测试完成,HyDE召回切片数:{len(hyde_context)}")
