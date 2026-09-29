"""
答案生成服务：拼装上下文并调用 LLM 生成回答（node_answer_output 节点的业务实现）。

上下文拼装策略：
- 按精排得分降序拼接切片，标注来源（file_title > parent_title）便于答案溯源
- 累计字符数达到 ANSWER_CONTEXT_MAX_CHARS 即截断，防止超出模型上下文窗口
- 无召回时返回固定兜底话术，不浪费一次 LLM 调用
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import ANSWER_CONTEXT_MAX_CHARS
from app.rag.query.query_rewrite_service import format_history_text
from app.shared.model.lm_utils import get_llm_client
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

# 无召回时的兜底话术（不浪费一次LLM调用）
NO_CONTEXT_ANSWER = "未在知识库中检索到与问题相关的内容，请尝试换个问法，或先导入相关文档。"


def build_context_text(rerank_context: list[dict], max_chars: int = ANSWER_CONTEXT_MAX_CHARS) -> str:
    """把精排后的切片拼装成带来源标注的上下文文本，超长截断。"""
    # 1.逐片拼接：来源标注 + 切片正文，累计超限即停（精排分越高越靠前，优先保留）
    parts: list[dict] = []
    total_chars = 0
    for hit in rerank_context:
        part = (
            f"【来源】{hit.get('file_title', '')} > {hit.get('parent_title', '')}\n"
            f"{hit.get('chunk_text', '')}"
        )
        if total_chars + len(part) > max_chars:
            logger.info(f"上下文已达上限{max_chars}字符,截断剩余{len(rerank_context) - len(parts)}个切片")
            break
        parts.append(part)
        total_chars += len(part)
    return "\n\n".join(parts)


@step_log("generate_answer")
def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """基于精排上下文生成最终答案，写回 state["answer"]。"""
    # 1.获取参数（问题用改写后的完整问题，指代已消解）
    question = (state.get("rewritten_query") or "").strip() or (state.get("query") or "").strip()
    history_text = format_history_text(state.get("history") or [])
    item_names_text = "、".join(state.get("item_names") or []) or "未识别"

    # 2.无召回直接兜底，不调用LLM
    context = build_context_text(state.get("rerank_context") or [])
    if not context:
        logger.warning("rerank_context为空,返回兜底话术,跳过LLM调用!")
        state["answer"] = NO_CONTEXT_ANSWER
        return state

    # 3.渲染提示词并调用LLM生成答案
    prompt = load_prompt(
        "answer_out",
        context=context,
        history=history_text,
        item_names=item_names_text,
        question=question,
    )
    llm = get_llm_client()
    response = llm.invoke(prompt)
    # 4.答案写回状态（strip去除首尾空白，防止污染前端展示）
    state["answer"] = str(response.content or "").strip()
    logger.info(f"答案生成完成,长度:{len(state['answer'])}字符")
    return state
