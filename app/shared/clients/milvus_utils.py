"""
Milvus 客户端工具模块，负责提供全局唯一的 MilvusClient 单例。

所有需要访问 Milvus 的网关/服务统一通过 get_milvus_client() 获取客户端，
避免重复建连；连接参数来自 app/shared/config/milvus_config.py。
"""
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from app.shared.config.milvus_config import milvus_config
from app.shared.runtime.logger import logger

# 模块级客户端缓存（单例）
_milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """
    获取全局唯一的 Milvus 客户端实例（懒加载单例）。

    :return: 初始化完成的 MilvusClient 实例
    :raise ValueError: MILVUS_URL 未配置时提前抛出
    :raise Exception: Milvus 连接失败时向上抛出，由调用方处理
    """
    global _milvus_client
    # 单例命中：已初始化则直接返回，避免重复建连
    if _milvus_client is not None:
        logger.debug("Milvus客户端单例已存在，直接返回实例")
        return _milvus_client

    # 入参合法性校验：拦截缺失的连接配置，提前抛出明确异常
    if not milvus_config.milvus_url:
        logger.error("MILVUS_URL未配置,无法创建Milvus客户端!请检查.env配置!!")
        raise ValueError("milvus_url为空!请在.env中配置MILVUS_URL后再访问Milvus!!")

    logger.info(f"开始初始化Milvus客户端,连接地址:{milvus_config.milvus_url}")
    try:
        _milvus_client = MilvusClient(uri=milvus_config.milvus_url)
    except Exception as e:
        logger.error(f"Milvus客户端初始化失败:{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理
    logger.success("Milvus客户端初始化成功!!")
    return _milvus_client


def create_hybrid_search_requests(query_embedding: dict, limit: int) -> list[AnnSearchRequest]:
    """
    把单条查询的混合向量组装成稠密+稀疏两路检索请求（AnnSearchRequest）。

    两路召回的分工：
    - dense_vector（稠密，HNSW索引）：负责语义相似，换说法也能召回
    - sparse_vector（稀疏，倒排索引）：负责精确词匹配，型号/错误码等关键词不吃亏

    :param query_embedding: 单条查询的混合向量 {"dense": [float], "sparse": {维度: 权重}}
    :param limit: 每路召回数量（两阶段检索的召回段，宁可多不可漏）
    :return: 稠密+稀疏两路的 AnnSearchRequest 列表
    """
    # 稠密路：HNSW 检索，ef 为搜索时候选队列长度（越大越准越慢）
    dense_request = AnnSearchRequest(
        data=[query_embedding["dense"]],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=limit,
    )
    # 稀疏路：倒排检索，无需额外搜索参数
    sparse_request = AnnSearchRequest(
        data=[query_embedding["sparse"]],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=limit,
    )
    return [dense_request, sparse_request]


def hybrid_search(
    client: MilvusClient,
    collection_name: str,
    requests: list[AnnSearchRequest],
    top_k: int,
    output_fields: list[str],
) -> list[dict]:
    """
    执行混合检索并用 RRFRanker 融合两路结果。

    RRF（Reciprocal Rank Fusion）按排名倒数融合：score = Σ 1/(k + rank_i)，
    k=60 为论文经典取值——k 越大各路权重越均衡；它只看排名不看原始分数，
    天然规避了稠密 IP 分数与稀疏词权重量纲不一致、无法直接相加的问题。

    :param client: Milvus 客户端实例
    :param collection_name: 目标集合名
    :param requests: 两路检索请求列表
    :param top_k: 融合后返回的切片数量
    :param output_fields: 需要随结果返回的业务字段
    :return: 融合排序后的切片列表（业务字段 + distance 融合得分）
    """
    # 混合检索：两路各查各的，RRF 在服务端按排名融合后统一返回 top_k 条
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=requests,
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=output_fields,
    )
    # 返回结构：每条查询一组结果，这里单查询取第一组
    hits = results[0] if results else []
    # 扁平化为业务字典，distance 为 RRF 融合得分（仅用于排序展示，非相似度）
    return [
        {
            "id": hit.get("id"),
            "chunk_text": hit.get("chunk_text", ""),
            "file_title": hit.get("file_title", ""),
            "parent_title": hit.get("parent_title", ""),
            "part": hit.get("part", 1),
            "distance": hit.get("distance"),
        }
        for hit in hits
    ]
