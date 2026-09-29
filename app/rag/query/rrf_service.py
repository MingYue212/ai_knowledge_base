"""
RRF 融合服务：主路检索与 HyDE 路检索结果做排名倒数融合（node_rrf 节点的业务实现）。

为什么用 RRF（Reciprocal Rank Fusion）：
- 主路（问题向量）与 HyDE 路（假设性回答向量）各自召回 Top K，
  两路的 distance 分数量纲不可比（不同检索文本算出的相似度不能直接相加）
- RRF 只看排名不看原始分数：score = Σ 1/(k + rank)，k=60 为论文经典取值，
  rank 从 1 起；排名越靠前贡献越大，k 越大各路权重越均衡
- 以 Milvus 主键 id 去重：同一切片被两路同时命中说明相关性高，
  分数累加后排名靠前；融合结果降序截断 SEARCH_TOP_K 写回 state["search_context"]，
  交给 node_rerank 做交叉编码器精排

防御性设计（降级不中断）：
- HyDE 路为空（LLM/检索失败是常态）—— 退化为仅主路结果
- 主路为空 —— 退化为仅 HyDE 路结果
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import SEARCH_TOP_K
from app.shared.runtime.logger import logger, step_log

# RRF 常数 k：论文经典取值 60，k 越大各路排名的权重差越小（融合越均衡）
RRF_K = 60


def _hit_key(hit: dict) -> tuple:
    """构造去重键：优先用 Milvus 主键 id；无 id 时退化为切片正文。"""
    hit_id = hit.get("id")
    if hit_id is not None:
        return ("id", hit_id)
    return ("text", hit.get("chunk_text", ""))


@step_log("fuse_search_results")
def fuse_search_results(state: QueryGraphState) -> QueryGraphState:
    """主路与 HyDE 路召回切片做 RRF 融合，结果写回 state["search_context"]。

    防御性校验：
    1. 单路为空 —— 退化为另一路（HyDE 路失败不拖垮主路）
    2. 两路全空 —— search_context 写空列表，由答案节点给兜底话术
    """
    main_hits: list[dict] = state.get("search_context") or []
    hyde_hits: list[dict] = state.get("hyde_context") or []

    # 1.单路为空：退化为另一路（拷贝列表，避免两字段引用同一份结果）
    if not main_hits or not hyde_hits:
        state["search_context"] = list(main_hits or hyde_hits)
        logger.warning(f"RRF融合退化为单路,主路{len(main_hits)}个,HyDE路{len(hyde_hits)}个")
        return state

    # 2.RRF打分：score = Σ 1/(k + rank)，rank从1起；同id多路命中分数累加
    pool: dict[tuple, dict] = {}
    for road_hits in (main_hits, hyde_hits):
        for rank, hit in enumerate(road_hits, start=1):
            key = _hit_key(hit)
            if key not in pool:
                # 拷贝切片，避免污染原始两路结果
                pool[key] = {**hit, "rrf_score": 0.0}
            pool[key]["rrf_score"] += 1.0 / (RRF_K + rank)

    # 3.按融合得分降序截断Top K，写回主路状态字段（下游node_rerank继续精排）
    fused = sorted(pool.values(), key=lambda hit: hit["rrf_score"], reverse=True)
    state["search_context"] = fused[:SEARCH_TOP_K]
    logger.info(
        f"RRF融合完成,主路{len(main_hits)}个+HyDE路{len(hyde_hits)}个,"
        f"去重后{len(pool)}个,保留Top{len(state['search_context'])}"
    )
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始fuse_search_results服务单元测试 =====")

    # 测试1：两路正常融合（切片2两路共同命中，RRF分数累加后应排最前）
    test_state = create_default_state(task_id="test_rrf_001", query="烫金机怎么使用？")
    test_state["search_context"] = [
        {"id": 1, "chunk_text": "切片A", "file_title": "手册", "parent_title": "步骤", "part": 1},
        {"id": 2, "chunk_text": "切片B", "file_title": "手册", "parent_title": "参数", "part": 1},
    ]
    test_state["hyde_context"] = [
        {"id": 2, "chunk_text": "切片B", "file_title": "手册", "parent_title": "参数", "part": 1},
        {"id": 3, "chunk_text": "切片C", "file_title": "手册", "parent_title": "维护", "part": 1},
    ]
    result_state = fuse_search_results(test_state)
    logger.info(f"融合结果Top3: {[hit['chunk_text'] for hit in result_state['search_context']]}")
    logger.info(f"融合得分: {[round(hit['rrf_score'], 4) for hit in result_state['search_context']]}")

    # 测试2：HyDE路为空，退化为主路结果
    test_state_2 = create_default_state(task_id="test_rrf_002", query="烫金机怎么使用？")
    test_state_2["search_context"] = [{"id": 1, "chunk_text": "切片A"}]
    test_state_2["hyde_context"] = []
    result_state_2 = fuse_search_results(test_state_2)
    logger.info(f"单路退化结果: {[hit['chunk_text'] for hit in result_state_2['search_context']]}")
