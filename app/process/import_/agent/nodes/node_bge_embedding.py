"""
向量化节点：为文档切块生成 BGE-M3 稠密+稀疏混合向量。

节点链位置：node_document_split → node_item_name_recognition → [node_bge_embedding] → node_import_milvus
职责：调用 embedding_service 为全部切块生成混合向量，写入 state["embedding_context"]。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.embedding_service import generate_chunk_embeddings
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量生成 (node_bge_embedding)
    为什么叫这个名字: 调用 BGE-M3 模型为切块生成稠密+稀疏混合向量。
    """
    add_running_task(state["task_id"], "node_bge_embedding")
    state = generate_chunk_embeddings(state)
    add_done_task(state["task_id"], "node_bge_embedding")
    return state


if __name__ == '__main__':
    from app.process.import_.agent.state import create_default_state
    from app.shared.runtime.logger import logger
    from app.shared.utils.path_util import PROJECT_ROOT
    import json
    import os

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 构造两个测试切块（内容很短，便于快速验证向量化链路）
    test_state = create_default_state(
        task_id="test_task_bge_001",
        local_file_path=os.path.join(str(PROJECT_ROOT), "test", "test.md"),
    )
    test_state["chunks"] = [
        {
            "title": "产品安全手册_概述",
            "content": "本产品适用于室内环境，使用前请仔细阅读安全手册。",
            "file_title": "产品安全手册",
            "parent_title": "产品安全手册",
            "part": 1,
        },
        {
            "title": "产品安全手册_注意事项",
            "content": "切勿在潮湿环境中使用本产品，避免触电风险。",
            "file_title": "产品安全手册",
            "parent_title": "产品安全手册",
            "part": 2,
        },
    ]

    logger.info("===== 开始node_bge_embedding节点单元测试 =====")
    result_state = node_bge_embedding(test_state)
    embedding_context = result_state.get("embedding_context") or []
    logger.info(
        f"本地测试完成 - 向量数量:{len(embedding_context)},"
        f"稠密维度:{len(embedding_context[0]['dense']) if embedding_context else 0}"
    )
    logger.info(f"测试结果: \n{json.dumps(embedding_context[:1], ensure_ascii=False)[:500]}")
