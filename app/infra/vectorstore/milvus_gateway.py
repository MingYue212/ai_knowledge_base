"""
Milvus 向量库网关：封装集合管理与混合向量（稠密+稀疏）写入。

所有 Milvus 访问统一经由本网关，业务层不直接接触 pymilvus；
集合不存在时按 schema 自动创建并构建索引（幂等，可重复调用）。
"""
from pymilvus import AnnSearchRequest, DataType, MilvusClient

from app.infra.config.providers import infra_config
from app.shared.clients.milvus_utils import get_milvus_client, hybrid_search
from app.shared.runtime.logger import logger

# 稠密向量索引参数：HNSW + IP（BGE-M3 向量已做 L2 归一化，IP 等价余弦相似度）
_DENSE_INDEX_PARAMS = {"M": 24, "efConstruction": 200}


class MilvusGateway:
    """Milvus 网关：客户端、集合名称与建集合/插入能力的统一入口。"""

    @property
    def client(self) -> MilvusClient:
        """
        获取 Milvus 客户端实例（懒加载单例，见 shared/clients/milvus_utils）。

        Returns:
            MilvusClient: 底层 Milvus 客户端对象。
        """
        return get_milvus_client()

    @property
    def chunks_collection(self) -> str:
        """
        获取文档切块集合名称。

        Returns:
            str: Milvus 中存放知识切块（含混合向量）的集合名。
        """
        return infra_config.milvus_config.chunks_collection

    @property
    def item_name_collection(self) -> str:
        """
        获取主体名称集合名称。

        Returns:
            str: Milvus 中存放主体名称向量的集合名。
        """
        return infra_config.milvus_config.item_name_collection

    def ensure_chunks_collection(self, dense_dim: int) -> None:
        """
        确保文档切块集合存在，不存在则创建（幂等，可重复调用）。

        schema：chunk_text/file_title/parent_title/part 业务字段
                + sparse_vector（稀疏）+ dense_vector（稠密）双向量字段。
        索引：稠密 HNSW(IP)、稀疏 SPARSE_INVERTED_INDEX(IP)，
              配合 BGE-M3 归一化向量做混合检索。

        Args:
            dense_dim: 稠密向量维度（BGE-M3 为 1024），按调用方实际向量维度传入。
        """
        collection_name = self.chunks_collection
        if self.client.has_collection(collection_name):
            logger.debug(f"集合[{collection_name}]已存在,跳过创建")
            return

        logger.info(f"集合[{collection_name}]不存在,开始创建!稠密维度={dense_dim},混合向量(bge-m3)schema")
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("chunk_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("file_title", DataType.VARCHAR, max_length=512)
        schema.add_field("parent_title", DataType.VARCHAR, max_length=512)
        schema.add_field("part", DataType.INT64)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="IP",
            params=_DENSE_INDEX_PARAMS,
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self.client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )
        logger.success(f"集合[{collection_name}]创建成功!!")

    def insert_chunks(self, entities: list[dict]) -> None:
        """
        批量插入文档切块实体（含稠密+稀疏混合向量）。

        Args:
            entities: 入库实体列表，字段与 chunks 集合 schema 一一对应。
        """
        # 防御性校验：空列表直接跳过，避免无意义的 Milvus 请求
        if not entities:
            logger.warning("insert_chunks的入参entities为空,跳过插入!")
            return
        result = self.client.insert(collection_name=self.chunks_collection, data=entities)
        logger.info(f"chunks集合插入完成,插入数量:{len(entities)},milvus返回:{result}")

    def ensure_item_name_collection(self, dense_dim: int) -> None:
        """
        确保主体名称集合存在，不存在则创建（幂等，可重复调用）。

        schema：item_name 业务字段 + dense_vector 稠密向量字段。

        Args:
            dense_dim: 稠密向量维度，按调用方实际向量维度传入。
        """
        collection_name = self.item_name_collection
        if self.client.has_collection(collection_name):
            logger.debug(f"集合[{collection_name}]已存在,跳过创建")
            return

        logger.info(f"集合[{collection_name}]不存在,开始创建!稠密维度={dense_dim}")
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("item_name", DataType.VARCHAR, max_length=512)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="IP",
            params=_DENSE_INDEX_PARAMS,
        )
        self.client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )
        logger.success(f"集合[{collection_name}]创建成功!!")

    def insert_item_name(self, entity: dict) -> None:
        """
        插入单条主体名称实体（向量化后的名称）。

        Args:
            entity: 入库实体，字段与 item_name 集合 schema 一一对应。
        """
        result = self.client.insert(collection_name=self.item_name_collection, data=[entity])
        logger.info(f"主体名称集合插入完成,milvus返回:{result}")

    def search_chunks(self, requests: list[AnnSearchRequest], top_k: int) -> list[dict]:
        """
        混合检索文档切块集合（稠密+稀疏两路召回，RRF 融合排序）。

        Args:
            requests: 由 create_hybrid_search_requests 组装的两路检索请求。
            top_k: 融合后返回的切片数量。

        Returns:
            list[dict]: 召回的切片列表（业务字段 + distance 融合得分）。
        """
        return hybrid_search(
            client=self.client,
            collection_name=self.chunks_collection,
            requests=requests,
            top_k=top_k,
            output_fields=["id", "chunk_text", "file_title", "parent_title", "part"],
        )

    # ---------- 切片管理（chunk 手动管理，业务入口见 rag/import_/chunk_manage_service） ----------

    _CHUNK_OUTPUT_FIELDS = ["id", "chunk_text", "file_title", "parent_title", "part"]

    def query_chunks(self, filter_expr: str, limit: int, offset: int = 0) -> list[dict]:
        """
        按过滤表达式查询文档切块（管理用途，不走向量检索）。

        Args:
            filter_expr: Milvus 过滤表达式；空串按全部处理（id >= 0）。
            limit: 返回条数上限。
            offset: 跳过条数（配合 limit 做分页）。

        Returns:
            list[dict]: 切片业务字段列表（含 id）。
        """
        expr = filter_expr.strip() or "id >= 0"
        rows = self.client.query(
            collection_name=self.chunks_collection,
            filter=expr,
            output_fields=self._CHUNK_OUTPUT_FIELDS,
            limit=limit,
            offset=offset,
        )
        logger.info(f"chunks集合条件查询完成,filter:{expr[:80]},返回{len(rows)}条")
        return [dict(row) for row in rows]

    def get_chunk(self, chunk_id: int) -> dict | None:
        """
        按主键取单个切片（含全部业务字段）。

        Args:
            chunk_id: Milvus 主键 id。

        Returns:
            dict | None: 切片实体；不存在时返回 None。
        """
        rows = self.client.get(
            collection_name=self.chunks_collection,
            ids=[chunk_id],
            output_fields=self._CHUNK_OUTPUT_FIELDS,
        )
        return dict(rows[0]) if rows else None

    def delete_chunks(self, chunk_ids: list[int]) -> None:
        """
        按主键批量删除切片。

        Args:
            chunk_ids: 待删除的 Milvus 主键 id 列表。
        """
        if not chunk_ids:
            logger.warning("delete_chunks的入参chunk_ids为空,跳过删除!")
            return
        result = self.client.delete(collection_name=self.chunks_collection, ids=chunk_ids)
        logger.info(f"chunks集合删除完成,ids:{chunk_ids},milvus返回:{result}")

    def insert_chunk(self, entity: dict) -> int | None:
        """
        插入单条切片实体，返回新生成的主键 id（auto_id 集合由 Milvus 生成）。

        Args:
            entity: 入库实体，字段与 chunks 集合 schema 一一对应（不含 id）。

        Returns:
            int | None: 新主键 id；Milvus 返回中取不到 ids 时为 None（调用方按需回查）。
        """
        result = self.client.insert(collection_name=self.chunks_collection, data=[entity])
        ids = result.get("ids") if isinstance(result, dict) else getattr(result, "ids", None)
        new_id = int(ids[0]) if ids else None
        logger.info(f"chunks集合单条插入完成,new_id:{new_id},milvus返回:{result}")
        return new_id


# 模块级单例，业务代码统一入口
milvus_gateway = MilvusGateway()
