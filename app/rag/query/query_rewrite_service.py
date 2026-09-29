"""
查询预处理服务：多轮指代消解 + 查询改写（node_item_name_confirm 节点的业务实现）。

为什么需要查询改写：
- 用户常在多轮对话中用代词提问（"这个怎么保修？"），原始query直接向量化会丢语义
- 先用 LLM 结合历史对话做指代消解，改写成包含商品名的独立完整问题，
  检索质量才能与首问对齐
- 顺带提取用户正在询问的商品名（item_names），供答案生成阶段引用
"""
import json

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import MAX_HISTORY_TURNS
from app.shared.model.lm_utils import get_llm_client
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log


def format_history_text(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """把历史对话格式化为提示词文本（超出轮数则只保留最近记录）。"""
    # 1.无历史直接返回占位文本，避免提示词出现空洞
    if not history:
        return "无"
    # 2.逐轮格式化：role 统一为 用户/助手，内容去空白
    lines: list[str] = []
    for turn in history[-max_turns:]:
        role = "用户" if str(turn.get("role", "")).lower() == "user" else "助手"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "无"


@step_log("confirm_item_names")
def confirm_item_names(state: QueryGraphState) -> QueryGraphState:
    """结合历史对话做指代消解与查询改写，写回 item_names / rewritten_query。

    防御性设计（降级不中断）：
    查询改写是"锦上添花"环节——LLM 调用失败或 JSON 解析失败时，
    回退原始问题继续检索，而不是让整个查询流程报错终止。
    """
    # 1.获取参数
    query: str = (state.get("query") or "").strip()
    history_text = format_history_text(state.get("history") or [])

    # 2.渲染提示词并调用LLM（json_mode强制返回可解析的json_object）
    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke(prompt)
        parsed = json.loads(str(response.content))
        # 3.解析商品名列表：非列表/空值全部兜底为空列表
        item_names = parsed.get("item_names") or []
        if not isinstance(item_names, list):
            item_names = []
        state["item_names"] = [str(name).strip() for name in item_names if str(name).strip()]
        # 4.解析改写问题：改写为空时回退原始问题
        rewritten_query = str(parsed.get("rewritten_query") or "").strip()
        state["rewritten_query"] = rewritten_query or query
        logger.info(f"查询改写完成,改写后:{state['rewritten_query']},识别商品:{state['item_names']}")
    except Exception as e:
        # 5.降级：改写失败不中断查询，回退原始问题
        logger.warning(f"查询改写失败,回退原始问题继续检索:{str(e)}")
        state["item_names"] = []
        state["rewritten_query"] = query
    return state
