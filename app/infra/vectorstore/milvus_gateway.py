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
            self.migrate_add_group_field(collection_name)
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
        # 知识组归属（权限过滤用）：nullable，存量行读出为 None，由回填端点归入默认组
        schema.add_field("group_id", DataType.INT64, nullable=True)

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
            self.migrate_add_group_field(collection_name)
            return

        logger.info(f"集合[{collection_name}]不存在,开始创建!稠密维度={dense_dim}")
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("item_name", DataType.VARCHAR, max_length=512)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("group_id", DataType.INT64, nullable=True)

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
            output_fields=["id", "chunk_text", "file_title", "parent_title", "part", "group_id"],
        )

    # ---------- 切片管理（chunk 手动管理，业务入口见 rag/import_/chunk_manage_service） ----------

    _CHUNK_OUTPUT_FIELDS = ["id", "chunk_text", "file_title", "parent_title", "part", "group_id"]

    def migrate_add_group_field(self, collection_name: str) -> bool:
        """
        给既有集合补 group_id 标量字段（幂等，pymilvus 2.6 的 add_collection_field）。

        Args:
            collection_name: 目标集合名。

        Returns:
            bool: 是否实际发生了迁移（已有该字段则为 False）。
        """
        desc = self.client.describe_collection(collection_name=collection_name)
        field_names = {field.get("name") for field in desc.get("fields", [])}
        if "group_id" in field_names:
            return False
        self.client.add_collection_field(
            collection_name=collection_name, field_name="group_id", data_type=DataType.INT64
        )
        logger.info(f"集合[{collection_name}]已补group_id字段(nullable,存量行为None)")
        return True

    def count_chunks_by_group(self, group_id: int) -> int:
        """
        统计某知识组下的切片数量（count(*) 服务端聚合）。

        Args:
            group_id: 知识组 id。

        Returns:
            int: 切片数量。
        """
        rows = self.client.query(
            collection_name=self.chunks_collection,
            filter=f"group_id == {int(group_id)}",
            output_fields=["count(*)"],
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0

    def backfill_group_ids(self, default_group_id: int) -> int:
        """
        把 group_id 为空（存量/迁移遗留）的切片回填进默认知识组。

        实现说明：query 取回全字段（含双向量）后在客户端筛出 group_id 为 None 的行，
        带主键 upsert 覆盖（auto_id 集合更新行的标准方式）；不依赖 IS NULL 表达式。

        Args:
            default_group_id: 默认知识组 id。

        Returns:
            int: 本次回填的切片条数（单次扫描上限 16384，超出会有告警日志）。
        """
        rows = self.client.query(
            collection_name=self.chunks_collection,
            filter="id >= 0",
            output_fields=["id", "chunk_text", "file_title", "parent_title", "part",
                           "sparse_vector", "dense_vector", "group_id"],
            limit=16384,
        )
        targets = []
        for row in rows:
            if row.get("group_id") is None:
                fixed = dict(row)
                fixed["group_id"] = int(default_group_id)
                targets.append(fixed)
        if len(rows) >= 16384:
            logger.warning("backfill_group_ids单次扫描达上限16384,可能存在未回填数据,请再次执行!")
        if not targets:
            logger.info("无需回填:所有切片均已有知识组归属")
            return 0
        self.client.upsert(collection_name=self.chunks_collection, data=targets)
        logger.info(f"存量切片回填完成,共{len(targets)}条归入默认组:{default_group_id}")
        return len(targets)

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
