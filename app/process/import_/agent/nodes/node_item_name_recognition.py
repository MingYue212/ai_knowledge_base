"""
主体名称识别节点：识别文档所述的商品/主体名称。

节点链位置：node_document_split → [node_item_name_recognition] → node_bge_embedding
当前为第0步直通实现：不调用 LLM，保持 state["item_name"] 为空，
后续步骤再接入 LLM 识别（见 item_name_service 的 TODO）。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.item_name_service import recognize_item_name
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体名称识别 (node_item_name_recognition)
    为什么叫这个名字: 从文档内容中识别商品/主体名称，供导入时建立主体索引。
    """
    add_running_task(state["task_id"], "node_item_name_recognition")
    state = recognize_item_name(state)
    add_done_task(state["task_id"], "node_item_name_recognition")
    return state


if __name__ == '__main__':
    from app.process.import_.agent.state import create_default_state
    from app.shared.runtime.logger import logger
    import json

    logger.info("===== 开始node_item_name_recognition节点单元测试 =====")

    # 构造带chunks的测试状态（当前直通实现不修改item_name，预期输出为空字符串）
    test_state = create_default_state(task_id="test_task_item_name_001")
    test_state["chunks"] = [
        {
            "title": "产品安全手册_概述",
            "content": "本产品适用于室内环境，使用前请仔细阅读安全手册。",
            "file_title": "产品安全手册",
            "parent_title": "产品安全手册",
            "part": 1,
        },
    ]

    result_state = node_item_name_recognition(test_state)
    logger.info(f"本地测试完成 - item_name:{result_state.get('item_name', '')!r}")
    logger.info(f"测试结果: \n{json.dumps({'item_name': result_state.get('item_name', '')}, ensure_ascii=False)}")
