"""
查询日志与知识缺失服务：普通问答的日志落库、帮助度反馈、缺失运营视图。

借鉴管理平台讲义模块十（知识缺失）的设计：
- 回答类型由引擎明确上报（0 正常 / 1 资料缺失拒答 / 2 权限拒答 / 3 澄清追问），不靠运营猜文案
- 当前引擎实现只产生 0/1 两种：答案命中知识库兜底话术（NO_CONTEXT_ANSWER）
  即记为"资料缺失拒答"；2/3 为预留枚举，等权限拒答与澄清追问链路接入后启用
- 反馈边界规则在网关层强制：仅正常回答可反馈、仅本人日志可反馈
- 知识缺失视图 = 拒答 ∪ 无帮助 的查询日志，供运营补资料（后续 FAQ 归簇会接管计数，见讲义模块九）
"""
from app.infra.persistence.sqlite_gateway import (
    ANSWER_TYPE_NO_KNOWLEDGE,
    ANSWER_TYPE_NORMAL,
    sqlite_gateway,
)
from app.process.query.agent.state import QueryGraphState
from app.rag.query.answer_service import NO_CONTEXT_ANSWER
from app.shared.runtime.logger import logger, step_log


def resolve_answer_type(answer: str) -> int:
    """由答案文本判定回答类型：命中知识库兜底话术即"资料缺失拒答"，否则正常回答。"""
    return ANSWER_TYPE_NO_KNOWLEDGE if (answer or "").strip() == NO_CONTEXT_ANSWER else ANSWER_TYPE_NORMAL


@step_log("record_query_log")
def record_query_log(state: QueryGraphState) -> int:
    """问答结束后落一条查询日志，返回日志 id（随 FINAL 事件下发给前端反馈按钮）。

    日志只记事实（问题/改写/答案/类型/身份），不做任何业务判断——判断留给查询时的过滤
    与运营视图（借鉴讲义"日志是事实表"的原则）。
    """
    answer = str(state.get("answer") or "")
    answer_type = resolve_answer_type(answer)
    log_id = sqlite_gateway.insert_query_log(
        task_id=str(state.get("task_id") or ""),
        user_id=state.get("user_id"),
        query=str(state.get("query") or ""),
        rewritten_query=str(state.get("rewritten_query") or ""),
        item_names=list(state.get("item_names") or []),
        answer=answer,
        answer_type=answer_type,
    )
    if answer_type == ANSWER_TYPE_NO_KNOWLEDGE:
        logger.warning(f"知识缺失记录:log:{log_id},query:{state.get('query')}")
    return log_id


def submit_feedback(log_id: int, feedback_status: int, user_id: int | None) -> None:
    """提交帮助度反馈（仅正常回答可反馈、仅本人日志可反馈，规则在网关层强制）。"""
    sqlite_gateway.set_feedback(log_id, feedback_status, user_id)


def list_gaps(page: int, page_size: int, gap_filter: str) -> dict:
    """知识缺失运营视图（拒答 ∪ 无帮助），网关分页结果直接透出。"""
    return sqlite_gateway.list_gaps(page, page_size, gap_filter)


def mark_gap_handled(log_id: int, handled: bool) -> None:
    """标记/撤销一条缺失记录的处理状态（补资料或已答复后由运营操作）。"""
    sqlite_gateway.set_gap_handled(log_id, handled)


def gap_summary() -> dict:
    """缺失概览计数：拒答数 / 无帮助数 / 待处理数。"""
    return sqlite_gateway.gap_summary()


if __name__ == '__main__':
    # 离线自测：临时库验证 answer_type 判定与日志链路（不依赖 LLM/Milvus）
    import tempfile
    from pathlib import Path

    import app.rag.query.query_log_service as svc
    from app.infra.persistence.sqlite_gateway import (
        FeedbackNotAllowedError,
        FEEDBACK_HELPFUL,
        FEEDBACK_UNHELPFUL,
        SqliteGateway,
    )

    svc.sqlite_gateway = SqliteGateway(db_path=Path(tempfile.mkdtemp()) / "t.db")

    assert svc.resolve_answer_type("正常答案") == 0
    assert svc.resolve_answer_type("  未在知识库中检索到与问题相关的内容，请尝试换个问法，或先导入相关文档。 ") == 1

    state = {"task_id": "t-log", "user_id": None, "query": "怎么保修", "rewritten_query": "烫金机怎么保修",
             "item_names": ["烫金机"], "answer": "正常答案"}
    log_id = svc.record_query_log(state)
    assert log_id == 1

    svc.submit_feedback(log_id, FEEDBACK_HELPFUL, None)
    try:
        refusal_state = {**state, "answer": "未在知识库中检索到与问题相关的内容，请尝试换个问法，或先导入相关文档。"}
        refusal_id = svc.record_query_log(refusal_state)
        svc.submit_feedback(refusal_id, FEEDBACK_UNHELPFUL, None)
        raise AssertionError("拒答应不可反馈")
    except FeedbackNotAllowedError:
        pass

    gaps = svc.list_gaps(1, 20, "all")
    assert gaps["total"] == 1 and gaps["items"][0]["answer_type"] == 1
    svc.mark_gap_handled(refusal_id, True)
    assert svc.gap_summary() == {"refusal_count": 1, "unhelpful_count": 0, "pending_count": 0}

    print("QUERY_LOG_SERVICE_SELFTEST_ALL_PASSED")
