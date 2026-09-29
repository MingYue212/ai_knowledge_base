"""
RRF 融合节点：主路与 HyDE 路检索结果排名融合。

节点链位置：node_search_embedding_hyde → [node_rrf] → node_rerank
职责：调用 rrf_service 将主路 search_context 与 HyDE 路 hyde_context
按 RRF 排名融合，融合结果写回 state["search_context"]。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query.rrf_service import fuse_search_results
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_rrf")
def node_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 倒排融合 (node_rrf)
    为什么叫这个名字: 两路召回只按排名做 RRF 融合，规避两路分数量纲不可比的问题。
    """
    add_running_task(state["task_id"], "node_rrf")
    state = fuse_search_results(state)
    add_done_task(state["task_id"], "node_rrf")
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始node_rrf节点单元测试 =====")
    test_state = create_default_state(task_id="test_rrf_node_001", query="烫金机怎么使用？")
    test_state["search_context"] = [
        {"id": 1, "chunk_text": "切片A", "file_title": "手册", "parent_title": "步骤", "part": 1},
        {"id": 2, "chunk_text": "切片B", "file_title": "手册", "parent_title": "参数", "part": 1},
    ]
    test_state["hyde_context"] = [
        {"id": 2, "chunk_text": "切片B", "file_title": "手册", "parent_title": "参数", "part": 1},
        {"id": 3, "chunk_text": "切片C", "file_title": "手册", "parent_title": "维护", "part": 1},
    ]
    result_state = node_rrf(test_state)
    search_context = result_state.get("search_context") or []
    logger.info(f"测试完成,融合后切片数:{len(search_context)}")
    for idx, hit in enumerate(search_context[:3], start=1):
        logger.info(f"Top{idx}: rrf_score={hit.get('rrf_score'):.4f}, {hit.get('chunk_text')}")
