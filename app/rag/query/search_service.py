"""
查询检索服务：查询向量化 + Milvus 混合检索（node_search_embedding 节点的业务实现）。

混合检索两路召回的分工：
- 稠密路（dense_vector，HNSW+IP）：语义相似，换个说法也能召回
- 稀疏路（sparse_vector，倒排+IP）：精确词匹配，型号/错误码等关键词不吃亏
两路结果由 Milvus 服务端用 RRFRanker 按排名融合（见 milvus_utils.hybrid_search）。
"""
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import SEARCH_TOP_K
from app.shared.clients.milvus_utils import build_group_filter_expr, create_hybrid_search_requests
from app.shared.model.embedding_utils import generate_query_embeddings
from app.shared.runtime.logger import logger, step_log


@step_log("search_chunks")
def search_chunks(state: QueryGraphState) -> QueryGraphState:
    """查询向量化 + 混合检索，召回切片写入 state["search_context"]。

    防御性校验：
    1. 检索词非空 —— 优先用改写后的问题（含指代消解结果），为空回退原始问题
    2. 召回为空 —— 属于正常业务状态（知识库无相关内容），记录告警不报错，
       由答案生成节点给出兜底话术
    """
    # 1.确定检索词：改写优先，回退原始query
    query: str = (state.get("rewritten_query") or "").strip() or (state.get("query") or "").strip()
    if not query:
        logger.error("query与rewritten_query均为空,检索无法进行,提前终止!!")
        raise ValueError("检索词为空!请检查上游节点是否正确写入query!!")

    # 2.查询侧向量化（encode_queries，与文档侧 encode_documents 对应）
    embeddings = generate_query_embeddings([query])
    query_embedding = {
        "dense": embeddings["dense"][0],
        "sparse": embeddings["sparse"][0],
    }

    # 3.组装稠密+稀疏两路检索请求，交由Milvus服务端RRF融合（带知识组权限过滤下推）
    group_expr = build_group_filter_expr(state.get("allowed_group_ids"))
    if group_expr:
        logger.info(f"知识组权限过滤已生效:{group_expr}")
    requests = create_hybrid_search_requests(query_embedding, limit=SEARCH_TOP_K, expr=group_expr)
    hits: list[dict] = milvus_gateway.search_chunks(requests, top_k=SEARCH_TOP_K)

    # 4.写回状态：空召回不中断（答案节点有兜底话术）
    state["search_context"] = hits
    if not hits:
        logger.warning(f"混合检索未召回任何切片!query:{query}")
    else:
        logger.info(f"混合检索完成,召回{len(hits)}个切片")
    return state
