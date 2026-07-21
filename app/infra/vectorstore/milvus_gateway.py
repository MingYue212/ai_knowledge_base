from typing import Any

from app.shared.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.infra.config.providers import infra_config


class MilvusGateway:

    @property
    def chunks_collection(self) -> str:
        """
        获取文档切块集合名称。

        Returns:
            str: Milvus 中存放知识切块的集合名。
        """
        return infra_config.milvus.chunks_collection

    @property
    def item_name_collection(self) -> str:
        """
        获取主体名称集合名称。

        Returns:
            str: Milvus 中存放主体名称向量的集合名。
        """
        return infra_config.milvus.item_name_collection

    @property
    def client(self):
        """
        获取 Milvus 客户端实例。

        Returns:
            Any: 底层 Milvus 客户端对象。
        """
        return get_milvus_client()
milvus_gateway = MilvusGateway()