from app.process.query.agent.state import QueryGraphState
from app.rag.query.query_rewrite_service import confirm_item_names
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    节点: 确认问题产品 (node_item_name_confirm)
    为什么叫这个名字: 结合历史对话做指代消解（"这个/它"指哪个商品），
    改写出包含商品名的独立完整问题，并提取商品名列表。
    改写失败自动降级回退原始问题，不中断查询链路。
    """
    add_running_task(state["task_id"], "node_item_name_confirm")
    state = confirm_item_names(state)
    add_done_task(state["task_id"], "node_item_name_confirm")
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始node_item_name_confirm节点单元测试 =====")
    # 注意：本测试需要 OPENAI_API_KEY / OPENAI_BASE_URL 已配置（LLM指代消解）

    # 测试1：无历史单轮问题（改写应等于或近似原问题）
    test_state_1 = create_default_state(task_id="test_confirm_001", query="烫金机怎么使用？")
    result_1 = node_item_name_confirm(test_state_1)
    logger.info(f"测试1改写后:{result_1['rewritten_query']},商品:{result_1['item_names']}")

    # 测试2：多轮指代（"这个"应消解为具体商品）
    test_state_2 = create_default_state(
        task_id="test_confirm_002",
        query="它的加热功率是多少？",
        history=[
            {"role": "user", "content": "苏泊尔烫金机多少钱？"},
            {"role": "assistant", "content": "苏泊尔烫金机售价约299元。"},
        ],
    )
    result_2 = node_item_name_confirm(test_state_2)
    logger.info(f"测试2改写后:{result_2['rewritten_query']},商品:{result_2['item_names']}")
