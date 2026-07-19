import base64
import re
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.infra.config.providers import infra_config
from app.infra.llm.providers import llm_provider
from app.infra.object_storage import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS, IMAGE_CONTEXT_SUB_CHARS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.shared.utils.rate_limit_utils import apply_api_rate_limit


def validate_and_data(state: ImportGraphState):
    # 1 非空校验
    md_path = state.get('md_path')
    if not md_path:
        logger.error(f"md_path变量为空,业务无法继续进行,提前终止!")
        raise ValueError(f"md_path变量为空,业务无法继续进行,提前终止!")
    # 2 文件存在校验
    md_path_obj: Path = Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"md_path变量为{md_path},但是没有具体的文件! 业务无法继续进行,提前终止!")
        raise FileNotFoundError(f"md_path变量为{md_path},但是没有具体的文件! 业务无法继续进行,提前终止!")
    # 3 获取md_content、md_image_dir_obj
    md_content = md_path_obj.read_text(encoding='utf-8')
    md_image_dir_obj: Path = md_path_obj.parent / 'images'
    # 4 写入状态
    state['md_content'] = md_content
    # 5 返回
    return md_path_obj, md_image_dir_obj, md_content


def scan_images(md_content: str, md_images_dir_obj: Path) -> list[tuple[str, str, tuple[str, str]]]:
    image_info_list = []
    for image_file_obj in md_images_dir_obj.iterdir():
        image_name: str = image_file_obj.name
        image_path: str = str(image_file_obj)
        if image_file_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            # 不是图片
            logger.warning(f"本次处理的文件名:{image_name},不是图片,跳过本次处理!!")
            continue
            # 是一张图片 , 图片名 -> md_content是否存在 ![](xx)
        reg = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        search_match = reg.search(md_content)
        if not search_match:
            # 为空,真没有匹配到
            logger.warning(f"{image_name}没有被md_content引用引用,跳过,直接下一次!!")
            continue
        # 找到了 这里有一张图 [!] [示意图](images/demo(1).png [)] 再来一张
        start = search_match.start()
        end = search_match.end()
        # todo 定义一个常量 IMAGE_CONTEXT_SUB_CHARS config.py
        # [ )
        pre_context = md_content[max(0, start - IMAGE_CONTEXT_SUB_CHARS):start]
        post_context = md_content[end:min(end + IMAGE_CONTEXT_SUB_CHARS, len(md_content))]
        logger.debug(
            f"{image_name}在md_content被引用,引用的位置:{start}:{end},截取的上文:{pre_context} , 下文:{post_context}")
        image_info_list.append(
            (
                image_name,
                image_path,
                (
                    pre_context,
                    post_context
                )
            )
        )
    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(image_info_list)}")
    return image_info_list


def summarize_images(image_info_list: list[tuple[str, str, tuple[str, str]]], root_folder: str) -> dict[str, str]:
    """
     调用视觉模型,识别图片的含义!!!
    :param image_info_list:
    :return:
    """
    summary_image_dict: dict[str, str] = {}
    # 1.准备模型对象
    vision_model = llm_provider.vision_model(vision_model_name=infra_config.lm_config.lv_model)
    # 2.循环image_info_list数据,获取每一张图片的信息
    for image_name, image_path, image_content in image_info_list:
        # 3.循环里,将图片的信息拼接成提示词
        # 视觉模型->初步知道-> 1. 图片 2.文本
        image_prompt_text = load_prompt("image_summary", root_folder=root_folder, image_content=image_content)
        image_path_obj: Path = Path(image_path)
        image_base64_str: str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_base64_str}"}
                },  # 图片
                {
                    "type": "text", "text": image_prompt_text
                }  # 文本
            ]
        )
        # 4.调用视觉模型获取结果
        # 调用链 chains
        chains = vision_model | StrOutputParser()
        # 添加范围内限制
        apply_api_rate_limit()

        image_summary = chains.invoke([message])
        # 5.结果拼接到字典数据中
        summary_image_dict[image_name] = image_summary
        logger.debug(f"完成:{image_name}的视觉识别,对应的含义:{image_summary}")
        # 6.循环外返回字典数据
    return summary_image_dict


def upload_images_get_url(image_info_list: list[tuple[str, str, tuple[str, str]]], stem: str) -> dict[str, str]:
    """
      上传文件,并获取文件的访问url地址!!
    :param image_info_list:
    :param stem:
    :return:
    """
    # 1.删除minio中对应的文件的所有图片
    minio_client = minio_gateway.minio_client
    # 先查询 [失败]  image_dir ->  /upload-images
    list_object = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        # /upload-images + / + 文件名
        # todo: 千万不能使用 /开头
        prefix=minio_gateway.image_dir[1:] + "/" + stem,
        recursive=True
    )
    delete_object_list = [DeleteObject(obj.object_name) for obj in list_object]
    errors = minio_client.remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_object_list
    )
    for error in errors:
        logger.warning(f"删除图片出现问题:{error}")
    # 2.重新上传本次对应的文件
    image_url_dict: dict[str, str] = {}
    for image_name, image_path, _ in image_info_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=minio_gateway.image_dir + "/" + stem + "/" + image_name,  # 对象会决定在minio的中显示 桶名/前缀/文件/图片名
                file_path=image_path,
                content_type=guess_type(image_name)[0]
            )
            url = minio_gateway.build_image_url(stem, image_name)
            image_url_dict[image_name] = url
            logger.debug(f"{image_name}已经完成上传,对应的地址为:{url}")
        except Exception as e:
            logger.warning(f"{image_name}上传失败,跳过,继续下一张图片传递!!")
            # 3.记录图片和对应的访问地址即可
        return image_url_dict


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    # 1 md文件校验以及path content image_dir获取
    md_path_obj, md_images_dir_obj, md_content = validate_and_data(state)
    # 2 校验是否有图片
    if (not md_images_dir_obj.is_dir()) or len(list(md_images_dir_obj.iterdir())) == 0:
        # 不存在或者是文件 -> true
        # 存在是文件夹  or  没有没有文件 -> false
        logger.info(f"{md_path_obj}对应的md,没有图片内容,无需后续处理,直接跳出!!")
        return state
    # 3 获取图片信息和上下文
    image_info_list: list[tuple[str, str, tuple[str, str]]] = scan_images(md_content, md_images_dir_obj)
    # 4. 调用视觉模型 image_info_list [图片名,图片地址,上下文] -> {图片名:总结内容}
    summary_image_dict: dict[str, str] = summarize_images(image_info_list, md_path_obj.stem)
    # 5. 将图片信息传递到minio的服务器,并获取图片和对应的网络地址
    # {image_name:http...}
    image_url_dict: dict[str, str] = upload_images_get_url(image_info_list, md_path_obj.stem)
    print(image_url_dict)
    return state
