"""
重排序节点：交叉编码器精排。

节点链位置：node_search_embedding → [node_rerank] → node_answer_output
职责：调用 rerank_service 对召回切片精排，取 Top N 写入 state["rerank_context"]。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query.rerank_service import rerank_chunks
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_rerank")
def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 重排序 (node_rerank)
    为什么叫这个名字: 用交叉编码器对召回结果精排，提升进入上下文的切片质量。
    """
    add_running_task(state["task_id"], "node_rerank")
    state = rerank_chunks(state)
    add_done_task(state["task_id"], "node_rerank")
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始node_rerank节点单元测试 =====")
    # 注意：本测试需要 BGE-RERANKER 模型可用（首次运行会下载模型）
    test_state = create_default_state(task_id="test_rerank_001", query="烫金机怎么使用？")
    test_state["search_context"] = [
        {
            "chunk_text": "烫金机使用前请预热5分钟，待温度稳定后再放入物料。",
            "file_title": "烫金机产品手册",
            "parent_title": "使用步骤",
            "part": 1,
            "distance": 0.8,
        },
        {
            "chunk_text": "烫金机额定功率2000W，支持温度调节。",
            "file_title": "烫金机产品手册",
            "parent_title": "技术参数",
            "part": 1,
            "distance": 0.7,
        },
    ]
    result_state = node_rerank(test_state)
    rerank_context = result_state.get("rerank_context") or []
    logger.info(f"测试完成,精排后切片数:{len(rerank_context)}")
    for idx, hit in enumerate(rerank_context, start=1):
        logger.info(f"Top{idx}: score={hit.get('rerank_score'):.4f}, {hit.get('parent_title')}")
