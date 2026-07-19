"""
Milvus 向量库配置，定义连接地址与集合名。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str

@dataclass
class MilvusConfig:
    """Milvus 连接与集合配置。"""
    milvus_url: str
    chunks_collection: str
    entity_name_collection: str
    item_name_collection: str

milvus_config = MilvusConfig(
    milvus_url=env_str("MILVUS_URL"),
    chunks_collection=env_str("CHUNKS_COLLECTION"),
    entity_name_collection=env_str("ENTITY_NAME_COLLECTION"),
    item_name_collection=env_str("ITEM_NAME_COLLECTION"),
)
