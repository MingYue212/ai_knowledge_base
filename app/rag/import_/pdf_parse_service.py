import shutil
import time
from pathlib import Path
from typing import Tuple

import requests

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_DOWNLOAD_TIMEOUT_SECONDS, MINERU_POLL_INTERVAL_SECONDS, \
    MINERU_POLL_TIMEOUT_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT


def validate_path_values(state: ImportGraphState) -> Tuple[Path, Path]:
    # 1.获取需要校验的路径
    local_dir: str = state.get('local_dir')
    pdf_path: str = state.get('pdf_path')
    # 2.校验：pdf_path不能为空，local_dir 可以为空，为空则指定默认位置
    if not pdf_path:
        logger.error(f"pdf_path地址为空,业务无法继续,提前终止!")
        raise ValueError(f"pdf_path地址为空,业务无法继续,提前终止!")
    if not local_dir:
        local_dir: Path = PROJECT_ROOT / 'output'
        logger.warning(f"local_dir为空,不影响业务正常进行,给与默认,默认值:{str(local_dir)}")
    # 3.转为Path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj: Path = Path(local_dir)
    # 4.校验文件或文件夹是否为空
    if not pdf_path_obj.is_file():
        logger.error(f"pdf_path地址为:{pdf_path},不存在或者不是文件,业务无法继续,提前终止!")
        raise ValueError(f"pdf_path地址为:{pdf_path},不存在或者不是文件,业务无法继续,提前终止!")
    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir地址为:{str(local_dir_obj)},不存在或者不是文件夹,创建对应文件夹,业务继续!")
        # parents=True -> 如果是多层也会创建  x/x/x/x  mkdir -P, exist_ok=True 存在不报错
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    # 5.返回结果
    return pdf_path_obj, local_dir_obj


def upload_pdf_and_poll(pdf_path_obj: Path) -> str:
    # 1.向minerU申请，获取上传文件地址
    token = infra_config.mineru_config.api_key
    url = f"{infra_config.mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}", "data_id": f"{pdf_path_obj.stem}"}
        ],
        "model_version": MINERU_MODEL_VERSION,
    }
    response = requests.post(
        url=url,
        headers=header,
        json=data,
        timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS
    )

    status_code = response.status_code
    if status_code != 200:
        logger.error(f"向:{url}申请文件上传地址,请求失败,状态为:{status_code},业务无法继续,提前终止!")
        raise RuntimeError(f"向:{url}申请文件上传地址,请求失败,状态为:{status_code},业务无法继续,提前终止!")

    response_dict: dict = response.json()
    code = response_dict.get('code')
    msg = response_dict.get('msg')
    if code != 0:
        logger.error(f"向:{url}申请文件上传地址,业务失败,code:{code},错误信息:{msg},业务无法继续,提前终止!")
        raise RuntimeError(f"向:{url}申请文件上传地址,业务失败,code:{code},错误信息:{msg},业务无法继续,提前终止!")

    upload_file_urls: list[str] = response_dict.get('data', {}).get('file_urls', [])
    batch_id = response_dict.get('data', {}).get('batch_id')

    if not upload_file_urls:
        logger.error(f"向:{url}申请文件上传地址,业务失败,没有返回上传地址,业务无法继续,提前终止!")
        raise RuntimeError(f"向:{url}申请文件上传地址,业务失败,没有返回上传地址,业务无法继续,提前终止!")
    if not batch_id:
        logger.error(f"向:{url}申请文件上传地址,业务失败,没有批量标识,后续无法获取解析结果,提前终止!")
        raise RuntimeError(f"向:{url}申请文件上传地址,业务失败,没有批量标识,后续无法获取解析结果,提前终止!")

    upload_file_url: str = upload_file_urls[0]
    logger.info(f"申请文件解析地址成功! batch_id:{batch_id},地址:{upload_file_url}")

    # 2.上传
    file_data = pdf_path_obj.read_bytes()
    with requests.Session() as session:
        session.trust_env = False
        upload_response = session.put(
            url=upload_file_url,
            data=file_data,
            timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS
        )
        upload_status_code = upload_response.status_code
        if upload_status_code != 200:
            logger.error(
                f"向:{upload_file_url}上传文件,网络失败! stats_code:{upload_status_code},业务无法继续,提前终止!")
            raise RuntimeError(
                f"向:{upload_file_url}上传文件,网络失败!stats_code:{upload_status_code},业务无法继续,提前终止!")
        logger.info(f"向指定:{upload_file_url}地址上传文件:{pdf_path_obj.name}上传成功!准备获取解析结果!!")

    # 3.轮询获取解析结果URL
    max_wait_time = MINERU_POLL_TIMEOUT_SECONDS
    poll_time = MINERU_POLL_INTERVAL_SECONDS
    start_time = time.time()

    while True:
        if time.time() - start_time > max_wait_time:
            logger.error(f"向batch_id:{batch_id}轮询获取返回结果,等待时间超时! 业务提前终止!!")
            raise TimeoutError(f"向batch_id:{batch_id}轮询获取返回结果,等待时间超时! 业务提前终止!!")
        # 向minerU batch_id获取解析结果
        poll_url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"
        try:
            poll_response = requests.get(
                url=poll_url,
                headers=header
            )
        except Exception as e:
            logger.warning(f"向batch_id:{batch_id}轮询获取返回结果,网络请求异常,等待后,重试!!")
            time.sleep(poll_time)
            continue

        poll_response_status_code = poll_response.status_code
        if poll_response_status_code != 200:
            if 500 <= poll_response_status_code < 600:
                logger.warning(
                    f"向batch_id:{batch_id}轮询获取返回结果,网络请求异常,状态码为:{poll_response_status_code},等待后,重试!!")
                time.sleep(poll_time)
                continue
            else:
                logger.error(
                    f"向batch_id:{batch_id}轮询获取返回结果,状态码为:{poll_response_status_code},错误无法修复! 业务提前终止!!")
                raise RuntimeError(
                    f"向batch_id:{batch_id}轮询获取返回结果,状态码为:{poll_response_status_code},错误无法修复! 业务提前终止!!")
        poll_response_dict = poll_response.json()
        poll_code = poll_response_dict.get('code', -1)
        extract_result = poll_response_dict.get('data', {}).get('extract_result', [])[0]

        if poll_code != 0:
            logger.error(
                f"向batch_id:{batch_id}轮询获取返回结果,业务状态:{poll_code},业务无法修复! 业务提前终止!!")
            raise RuntimeError(
                f"向batch_id:{batch_id}轮询获取返回结果,业务状态:{poll_code},业务无法修复! 业务提前终止!!")
        poll_state = extract_result.get('state')

        if poll_state == "done":
            full_zip_url = extract_result.get("full_zip_url")
            full_zip_url = extract_result.get("full_zip_url")
            logger.info(f"基于:{pdf_path_obj.name}解析已经完成,对应下载地址为:{full_zip_url}")
            return full_zip_url
        elif poll_state == "failed":
            logger.error(
                f"向batch_id:{batch_id}轮询获取返回结果,文件解析失败!业务提前终止!!")
            raise RuntimeError(
                f"向batch_id:{batch_id}轮询获取返回结果,文件解析失败!业务提前终止!!")
        else:
            logger.warning(
                f"向batch_id:{batch_id}轮询获取返回结果,没有完成解析,等待后,重试!!")
            time.sleep(poll_time)
            continue


def download_and_extract_markdown(zip_download_url: str, local_dir_obj: Path, stem: str) -> Path:
    # 1.从获取的URL下载压缩文件
    response = requests.get(
        url=zip_download_url,
        timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        logger.error(f"{zip_download_url}地址下载资源报错!业务无法继续进行,提前终止!")
        raise RuntimeError(f"{zip_download_url}地址下载资源报错!业务无法继续进行,提前终止!")
    zip_file_obj: Path = local_dir_obj / f"{stem}.zip"
    zip_file_obj.write_bytes(data=response.content)

    # 2.创建要解压到的目录地址并解压
    zip_dir_obj: Path = local_dir_obj / stem
    if zip_dir_obj.is_dir():
        shutil.rmtree(zip_dir_obj)
    zip_dir_obj.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_file_obj, zip_dir_obj)

    # 3.检查是否解压出md文件，有则重命名
    md_file_list: list[Path] = list(zip_dir_obj.rglob("*.md"))
    if not md_file_list:
        logger.error(f"{zip_download_url}地址下载成功过,解压完成,但是没有md文件!业务无法继续进行,提前终止!")
        raise RuntimeError(f"{zip_download_url}地址下载成功过,解压完成,但是没有md文件!业务无法继续进行,提前终止!")
    for md_file_obj in md_file_list:
        if md_file_obj.stem == stem:
            logger.info(f"文件下载成功,解压成功!直接返回对应地址:{md_file_obj}")
            return md_file_obj
    full_md_file_obj: Path = None
    for md_file_obj in md_file_list:
        if md_file_obj.stem == "full":
            full_md_file_obj = md_file_obj
            logger.info(f"文件下载成功,解压成功!直接返回对应地址:{md_file_obj}")
            break
    full_md_file_obj.rename(full_md_file_obj.with_stem(f"{stem}.md"))
    if full_md_file_obj is None:
        raise RuntimeError("解压后的文件中未找到 full.md")
    return full_md_file_obj


def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    # 1.校验
    pdf_path_obj, local_dir_obj = validate_path_values(state)
    # 2.向minerU传PDF并得到md文件的下载URL
    zip_download_url: str = upload_pdf_and_poll(pdf_path_obj)
    # 3.下载、解压md文件
    md_path_obj: Path = download_and_extract_markdown(zip_download_url, local_dir_obj, pdf_path_obj.stem)
    # 4.把md_path更新到state
    state['md_path'] = str(md_path_obj)
    return state
