"""
配置统一出口：重新导出各领域配置类与单例，外部只需 import 本包即可使用。
"""
from app.shared.config.embedding_config import embedding_config, EmbeddingConfig
from app.shared.config.lm_config import lm_config, LLMConfig
from app.shared.config.bailian_mcp_config import mcp_config, McpConfig
from app.shared.config.milvus_config import milvus_config, MilvusConfig
from app.shared.config.mineru_config import mineru_config, MinerUConfig
from app.shared.config.minio_config import minio_config, MinIOConfig
from app.shared.config.reranker_config import reranker_config, RerankerConfig
from app.shared.config.settings_config import settings, AppSettings
