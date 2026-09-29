"""
入库节点：将文档切块与混合向量写入 Milvus。

节点链位置：node_bge_embedding → [node_import_milvus] → END
职责：调用 milvus_import_service 完成建集合（幂等）+ 实体批量插入；
若 state["item_name"] 非空，同时把主体名称向量化后写入主体名称集合。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.milvus_import_service import import_chunks_to_milvus
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 把切块实体与混合向量导入 Milvus，完成知识入库。
    """
    add_running_task(state["task_id"], "node_import_milvus")
    state = import_chunks_to_milvus(state)
    add_done_task(state["task_id"], "node_import_milvus")
    return state


if __name__ == '__main__':
    from app.infra.vectorstore.milvus_gateway import milvus_gateway
    from app.shared.runtime.logger import logger
    from app.shared.utils.path_util import PROJECT_ROOT
    import json

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 构造测试切块与假向量（稠密1024维/稀疏少量维度），验证建集合与插入链路
    # 注意：本测试需要本地 Milvus 已启动（MILVUS_URL）
    test_state = {
        "task_id": "test_task_milvus_001",
        "chunks": [
            {
                "title": "产品安全手册_概述",
                "content": "本产品适用于室内环境，使用前请仔细阅读安全手册。",
                "file_title": "产品安全手册",
                "parent_title": "产品安全手册",
                "part": 1,
            },
        ],
        "embedding_context": [
            {
                "dense": [0.01] * 1024,
                "sparse": {3: 0.5, 7: 0.25, 11: 0.25},
            },
        ],
    }

    logger.info("===== 开始node_import_milvus节点单元测试 =====")
    result_state = node_import_milvus(test_state)
    logger.info(
        f"本地测试完成 - 入库切块数:{len(result_state.get('chunks', []))},"
        f"chunks集合:{milvus_gateway.chunks_collection}"
    )
    logger.info("本地测试结果如下:")
    logger.info(f"chunks_count:{len(result_state.get('chunks', []))}")
