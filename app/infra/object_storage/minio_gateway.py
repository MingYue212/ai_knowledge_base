"""MinIO 对象存储网关。

封装 MinIO 的桶操作与图片 URL 拼接，供 node_md_img 节点使用：
Markdown 解析出的本地图片上传 MinIO 后，通过 build_image_url 生成访问地址，
替换 md_content 中的图片引用，使 Milvus 入库内容引用的是远程可访问 URL。
"""
from app.infra.config.providers import infra_config
from app.shared.clients.minio_utils import get_minio_client


class MinIOGateway:
    """MinIO 操作门面，属性透传配置，方法拼接图片外网 URL。"""

    @property
    def bucket_name(self):
        """目标桶名，来自 .env 中的 MINIO_BUCKET_NAME。"""
        return infra_config.minio_config.bucket_name

    @property
    def image_dir(self):
        """图片存储子目录前缀，如 /kb-images。"""
        return infra_config.minio_config.minio_img_dir

    @property
    def minio_client(self):
        """惰性获取 MinIO 客户端单例。"""
        return get_minio_client()

    def build_image_url(self, file_name: str, object_name: str) -> str:
        """根据文档名和图片文件名拼接 MinIO 访问 URL。

        一个文档（如 hak180产品安全手册）可能含多张图片，
        file_name 作为目录名隔离不同文档，object_name 为图片文件名。

        URL 结构：{protocol}{endpoint}/{bucket_name}/{image_dir}/{file_name}/{object_name}
        示例：http://127.0.0.1:9000/enterprise-rag/kb-images/hak180产品安全手册/screenshot_01.jpg

        Args:
            file_name: 文档名，作为图片的父目录（如 hak180产品安全手册）
            object_name: 图片文件名（如 screenshot_01.jpg）
        """
        protocol = "https://" if infra_config.minio_config.minio_secure else "http://"
        return (
            f"{protocol}{infra_config.minio_config.endpoint}/"
            f"{infra_config.minio_config.bucket_name}"
            f"{infra_config.minio_config.minio_img_dir}/"
            f"{file_name}/"
            f"{object_name}"
        )


minio_gateway = MinIOGateway()
