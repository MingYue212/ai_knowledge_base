"""
重排序服务：交叉编码器精排（node_rerank 节点的业务实现）。

两阶段检索（retrieve-then-rerank）的核心思想：
- 召回段用双编码器（query与文档各自编码再算相似度）：文档向量可离线入库，
  检索快，能扫全库，但精度有限
- 精排段用交叉编码器（BGE-reranker，query+文档拼接后联合编码逐对打分）：
  精度高，但每对都要过一遍模型，只能对少量候选做
- 所以先混合检索取 Top K，再精排取 Top N 进入答案上下文
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_TOP_N
from app.shared.model.reranker_utils import get_reranker_model
from app.shared.runtime.logger import logger, step_log


@step_log("rerank_chunks")
def rerank_chunks(state: QueryGraphState) -> QueryGraphState:
    """对召回切片做交叉编码器精排，取 Top N 写回 state["rerank_context"]。

    防御性校验：
    1. 召回为空 —— 直接返回空精排结果（上游可能本就没召回），不报错
    2. compute_score 单条输入返回 float —— 统一包装成列表处理
    """
    # 1.获取参数：精排用与检索相同的查询词（改写优先）
    query: str = (state.get("rewritten_query") or "").strip() or (state.get("query") or "").strip()
    candidates: list[dict] = state.get("search_context") or []
    if not candidates:
        logger.warning("search_context为空,跳过精排,rerank_context置空!")
        state["rerank_context"] = []
        return state

    # 2.逐对打分：交叉编码器输入为 [query, 文档] 对
    pairs = [[query, hit.get("chunk_text", "")] for hit in candidates]
    scores = get_reranker_model().compute_score(pairs, normalize=True)
    # 单条输入时FlagEmbedding返回float，统一包装为列表
    if not isinstance(scores, list):
        scores = [scores]

    # 3.得分写回并按精排得分降序排序
    for hit, score in zip(candidates, scores):
        hit["rerank_score"] = float(score)
    ranked = sorted(candidates, key=lambda hit: hit["rerank_score"], reverse=True)

    # 4.截取Top N写回状态
    state["rerank_context"] = ranked[:RERANK_TOP_N]
    logger.info(
        f"精排完成,候选{len(candidates)}个,保留Top{len(state['rerank_context'])},"
        f"最高分:{state['rerank_context'][0]['rerank_score']:.4f}"
    )
    return state
