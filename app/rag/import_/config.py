"""
导入流程常量配置：MinerU 模型版本、轮询与下载超时参数。
"""
# MinerU 模型版本配置（vlm = 视觉语言模型，适合PDF/图片高精度解析）
MINERU_MODEL_VERSION = "vlm" #[p.. -> 内置的工作流, vlm -> pdfppt图片, html -> html文档]
# MinerU 任务轮询最大超时时间（单位：秒），超过则判定任务失败
# 600 秒 = 10 分钟，为复杂 PDF 的 AI 解析预留足够轮询时间
MINERU_POLL_TIMEOUT_SECONDS = 600
# MinerU 任务轮询间隔时间（单位：秒），每隔多久查询一次任务状态
MINERU_POLL_INTERVAL_SECONDS = 3
# MinerU 文件下载超时时间（单位：秒），下载文件超过此时长则中断
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 300


# 图片常见后缀名
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 截取图片的上下文的长度
IMAGE_CONTEXT_SUB_CHARS = 100