"""基础设施配置聚合门面。

各子模块的配置对象分散在 shared/config/ 下各自管理，
本模块通过 InfraConfig 提供唯一的聚合入口。
所有需要配置的业务代码统一 import infra_config 即可，
无需逐个关心每个子配置从哪里来。
"""
from app.shared.config.embedding_config import embedding_config, EmbeddingConfig
from app.shared.config.lm_config import lm_config, LLMConfig
from app.shared.config.bailian_mcp_config import mcp_config, McpConfig
from app.shared.config.milvus_config import milvus_config, MilvusConfig
from app.shared.config.mineru_config import mineru_config, MinerUConfig
from app.shared.config.minio_config import minio_config, MinIOConfig
from app.shared.config.reranker_config import reranker_config, RerankerConfig
from app.shared.config.settings_config import settings, AppSettings

from dataclasses import dataclass, field


@dataclass
class InfraConfig:
    """基础设施配置聚合根。

    使用 default_factory + lambda 实现惰性引用：
    每个字段指向 shared/config/ 下预初始化的模块级单例，
    不会在 InfraConfig 实例化时重复创建配置对象。
    调用方式：infra_config.mineru_config.base_url
    """
    embedding_config: EmbeddingConfig = field(default_factory=lambda: embedding_config)
    lm_config: LLMConfig = field(default_factory=lambda: lm_config)
    mcp_config: McpConfig = field(default_factory=lambda: mcp_config)
    milvus_config: MilvusConfig = field(default_factory=lambda: milvus_config)
    mineru_config: MinerUConfig = field(default_factory=lambda: mineru_config)
    minio_config: MinIOConfig = field(default_factory=lambda: minio_config)
    reranker_config: RerankerConfig = field(default_factory=lambda: reranker_config)
    settings: AppSettings = field(default_factory=lambda: settings)


# 模块级单例，业务代码统一入口
infra_config = InfraConfig()
