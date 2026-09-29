"""
答案生成节点：拼装上下文调用 LLM 生成回答。

节点链位置：node_rerank → [node_answer_output] → END
职责：调用 answer_service 生成最终答案写入 state["answer"]；
无召回时返回兜底话术，不浪费一次 LLM 调用。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query.answer_service import generate_answer
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 生成答案 (node_answer_output)
    为什么叫这个名字: 基于精排上下文生成最终回答，是查询链路的出口节点。
    """
    add_running_task(state["task_id"], "node_answer_output")
    state = generate_answer(state)
    add_done_task(state["task_id"], "node_answer_output")
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始node_answer_output节点单元测试 =====")

    # 测试1：无召回（预期返回兜底话术，不调用LLM）
    test_state_1 = create_default_state(task_id="test_answer_001", query="烫金机怎么用？")
    result_1 = node_answer_output(test_state_1)
    logger.info(f"测试1(无召回)答案:{result_1.get('answer')}")

    # 测试2：有召回（需要 OPENAI_API_KEY / OPENAI_BASE_URL 已配置）
    test_state_2 = create_default_state(task_id="test_answer_002", query="烫金机的加热功率是多少？")
    test_state_2["rerank_context"] = [
        {
            "chunk_text": "烫金机额定加热功率为2000W，支持温度调节。",
            "file_title": "烫金机产品手册",
            "parent_title": "技术参数",
        },
    ]
    result_2 = node_answer_output(test_state_2)
    logger.info(f"测试2(有召回)答案:{result_2.get('answer')}")
