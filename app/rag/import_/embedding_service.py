"""
文档切块向量化服务：调用 BGE-M3 生成稠密+稀疏混合向量。

输入 ImportGraphState.chunks（split_service 产出的切块列表），
输出写入 state["embedding_context"]，与 chunks 一一对应：
每项 {"dense": [float, ...], "sparse": {维度: 权重}}。
"""
from app.process.import_.agent.state import ImportGraphState
from app.shared.model.embedding_utils import generate_embeddings
from app.shared.runtime.logger import logger, step_log


@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """为全部切块生成混合向量并写回状态。

    防御性校验：
    1. chunks 非空 —— 上游切分失败时提前终止，避免空跑向量模型
    2. 每个切块的 content 非空 —— 空文本向量化无意义且可能产生脏向量
    3. 向量数量与切块数量一致 —— 拦截向量模型返回异常，避免入库错位
    """
    # 1.获取参数
    chunks: list[dict] = state.get("chunks") or []
    if not chunks:
        logger.error("chunks为空!文档切分节点未产出任何切块,向量化无法进行,提前终止!!")
        raise ValueError("state['chunks']为空!请先执行node_document_split节点完成文档切分!!")

    # 2.组装待向量化文本（与chunks顺序一一对应）
    texts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        content = (chunk.get("content") or "").strip()
        if not content:
            logger.error(f"第{idx}个切块的content为空,无法向量化!请检查split_service的切分结果!!")
            raise ValueError(f"chunks[{idx - 1}]['content']为空!业务无法继续进行,提前终止!!")
        texts.append(content)

    # 3.批量生成混合向量（模型单例内部加载，一次调用完成全部文本）
    logger.info(f"开始为{len(texts)}个切块生成BGE-M3混合向量")
    embeddings = generate_embeddings(texts)

    # 4.结果校验：向量数量必须与切块数量一致，否则入库会错位
    dense_vectors = embeddings.get("dense") or []
    sparse_vectors = embeddings.get("sparse") or []
    if len(dense_vectors) != len(texts) or len(sparse_vectors) != len(texts):
        logger.error(
            f"向量数量与切块数量不一致!chunks={len(texts)},"
            f"dense={len(dense_vectors)},sparse={len(sparse_vectors)}"
        )
        raise ValueError("生成的向量数量与切块数量不一致!向量模型返回异常,提前终止!!")

    # 5.写回state，与chunks按索引一一对应，供node_import_milvus组装实体入库
    state["embedding_context"] = [
        {"dense": dense, "sparse": sparse}
        for dense, sparse in zip(dense_vectors, sparse_vectors)
    ]
    logger.info(f"完成{len(state['embedding_context'])}个切块的混合向量化!")
    return state
