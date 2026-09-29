"""
HyDE 检索服务：假设性文档增强检索（node_search_embedding_hyde 节点的业务实现）。

为什么需要 HyDE（Hypothetical Document Embeddings）：
- 向量检索是"以文搜文"——问题与答案的语义空间距离 < 答案与答案的距离，
  用"问题"直接检索往往不如用"假设性答案"检索命中率高
- 先让 LLM 根据改写后的问题生成一段假设性回答（不需要真实正确，
  只要主题一致、术语相近），再用这段回答去向量库检索
- 假设性回答与真实文档（手册正文）的分布更接近，召回的切片更适合回答问题
- 主路（node_search_embedding）用问题检索负责"稳"，HyDE 路负责"补"，
  两路结果由 node_rrf 做排名融合（见 rrf_service.fuse_search_results）

防御性设计（降级不中断）：
- HyDE 是"锦上添花"的增强路——LLM 调用失败、假设性回答为空、检索异常时，
  hyde_context 置空返回，主路检索结果不受任何影响
"""
from langchain_core.messages import HumanMessage

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import SEARCH_TOP_K
from app.shared.clients.milvus_utils import create_hybrid_search_requests
from app.shared.model.embedding_utils import generate_query_embeddings
from app.shared.model.lm_utils import get_llm_client
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log


@step_log("search_by_hyde")
def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    """生成假设性回答并以文搜文，召回切片写入 state["hyde_context"]。

    防御性校验：
    1. 检索词为空 —— 与主路检索同源（rewritten_query 回退 query），为空直接置空返回
    2. 任何环节失败 —— hyde_context 置空，不抛异常（主路检索不受影响）
    """
    # 1.确定检索词：与主路检索同源（改写优先，回退原始query）
    query: str = (state.get("rewritten_query") or "").strip() or (state.get("query") or "").strip()
    if not query:
        logger.warning("query与rewritten_query均为空,HyDE检索跳过,hyde_context置空!")
        state["hyde_context"] = []
        return state

    try:
        # 2.渲染提示词并生成假设性回答（回答不需要真实正确，只要主题一致、术语相近）
        prompt = load_prompt("hyde_prompt", rewritten_query=query)
        response = get_llm_client().invoke([HumanMessage(content=prompt)])
        hyde_answer = str(response.content or "").strip()
        if not hyde_answer:
            logger.warning("HyDE假设性回答为空,跳过HyDE检索,hyde_context置空!")
            state["hyde_context"] = []
            return state
        state["hyde_answer"] = hyde_answer
        logger.info(f"HyDE假设性回答生成完成,长度:{len(hyde_answer)}字符")

        # 3.假设性回答向量化（encode_queries，与主路检索同侧编码）
        embeddings = generate_query_embeddings([hyde_answer])
        hyde_embedding = {
            "dense": embeddings["dense"][0],
            "sparse": embeddings["sparse"][0],
        }

        # 4.组装稠密+稀疏两路请求，以"假设性回答"为检索文本召回切片
        requests = create_hybrid_search_requests(hyde_embedding, limit=SEARCH_TOP_K)
        hits: list[dict] = milvus_gateway.search_chunks(requests, top_k=SEARCH_TOP_K)

        # 5.写回状态：空召回不中断（RRF融合时自动退化为仅主路）
        state["hyde_context"] = hits
        if not hits:
            logger.warning("HyDE检索未召回任何切片,hyde_context置空!")
        else:
            logger.info(f"HyDE检索完成,召回{len(hits)}个切片")
    except Exception as e:
        logger.warning(f"HyDE检索失败,hyde_context置空,主路检索不受影响:{str(e)}")
        state["hyde_context"] = []
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始search_by_hyde服务单元测试 =====")
    # 注意：本测试需要 LLM API、Milvus（MILVUS_URL）与 BGE-M3 模型均可用
    test_state = create_default_state(task_id="test_hyde_001", query="烫金机怎么使用？")
    result_state = search_by_hyde(test_state)
    hyde_context = result_state.get("hyde_context") or []
    logger.info(f"测试完成,HyDE召回切片数:{len(hyde_context)}")
    logger.info(f"假设性回答预览:{(result_state.get('hyde_answer') or '')[:80]}")
