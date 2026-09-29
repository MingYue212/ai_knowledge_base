from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_query_entry")
def node_query_entry(state: QueryGraphState) -> QueryGraphState:
    """
    入口节点：校验用户问题非空。

    为什么叫这个名字: 查询链路的第一个节点，负责拦截空问题，提前终止。
    """
    add_running_task(state["task_id"], "node_query_entry")

    # 1.获取参数
    query: str = (state.get("query") or "").strip()

    # 2.防御性校验：问题为空则查询链路无法进行，提前终止
    if not query:
        logger.error("query为空!查询链路无法进行,提前终止!!")
        raise ValueError("query为空!请检查API层是否正确传入用户问题!!")

    # 3.写回状态（去首尾空白，防止污染后续检索）
    state["query"] = query
    add_done_task(state["task_id"], "node_query_entry")
    return state


if __name__ == '__main__':
    from app.process.query.agent.state import create_default_state
    from app.shared.runtime.logger import logger

    logger.info("===== 开始node_query_entry节点单元测试 =====")

    # 测试1：空问题（预期抛ValueError）
    try:
        node_query_entry(create_default_state(task_id="test_query_001", query="   "))
    except ValueError as e:
        logger.info(f"测试1通过,空问题被拦截:{e}")

    # 测试2：正常问题
    result_state = node_query_entry(create_default_state(task_id="test_query_002", query="  烫金机怎么使用？  "))
    logger.info(f"测试2通过,strip后的问题:{result_state['query']}")
