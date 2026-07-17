"""
PDF 解析服务：调用 MinerU API 上传 PDF → 轮询等待解析完成 → 下载解压 Markdown。
"""
import shutil
import time
from pathlib import Path
from typing import Tuple

import requests

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_DOWNLOAD_TIMEOUT_SECONDS, MINERU_POLL_INTERVAL_SECONDS, \
    MINERU_POLL_TIMEOUT_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log


@step_log("validate_path_values")
def validate_path_values(state: ImportGraphState) -> Tuple[Path, Path]:
    """校验 pdf_path 和 local_dir，确保输入合法、输出目录存在。

    此函数可被脱离 LangGraph 图单独调用（例如命令行重解析脚本），
    因此不依赖 node_entry 的前置校验，自行兜底所有输入检查。
    local_dir 允许为空——此时默认写到项目根下的 output/ 目录。
    """
    local_dir: str = state.get('local_dir')
    pdf_path: str = state.get('pdf_path')
    if not pdf_path:
        logger.error(f"pdf_path地址为空,业务无法继续,提前终止!")
        raise ValueError(f"pdf_path地址为空,业务无法继续,提前终止!")
    if not local_dir:
        local_dir: Path = PROJECT_ROOT / 'output'
        logger.warning(f"local_dir为空,不影响业务正常进行,给与默认,默认值:{str(local_dir)}")
    pdf_path_obj = Path(pdf_path)
    local_dir_obj: Path = Path(local_dir)
    if not pdf_path_obj.is_file():
        logger.error(f"pdf_path地址为:{pdf_path},不存在或者不是文件,业务无法继续,提前终止!")
        raise ValueError(f"pdf_path地址为:{pdf_path},不存在或者不是文件,业务无法继续,提前终止!")
    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir地址为:{str(local_dir_obj)},不存在或者不是文件夹,创建对应文件夹,业务继续!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj


@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj: Path) -> str:
    """将 PDF 上传至 MinerU 并轮询等待解析完成。

    MinerU 异步解析流程：申请上传地址 → PUT 文件 → 轮询 /extract-results
    轮询状态机有三种出口：done（返回 zip 下载地址）、failed（抛异常）、processing（sleep 后继续）

    Returns:
        解析完成后的 zip 包下载地址
    """
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

    # 上传文件，关闭系统代理以避免企业环境代理干扰
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

    max_wait_time = MINERU_POLL_TIMEOUT_SECONDS
    poll_time = MINERU_POLL_INTERVAL_SECONDS
    start_time = time.time()

    while True:
        if time.time() - start_time > max_wait_time:
            logger.error(f"向batch_id:{batch_id}轮询获取返回结果,等待时间超时! 业务提前终止!!")
            raise TimeoutError(f"向batch_id:{batch_id}轮询获取返回结果,等待时间超时! 业务提前终止!!")
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

        if poll_code != 0:
            logger.error(
                f"向batch_id:{batch_id}轮询获取返回结果,业务状态:{poll_code},业务无法修复! 业务提前终止!!")
            raise RuntimeError(
                f"向batch_id:{batch_id}轮询获取返回结果,业务状态:{poll_code},业务无法修复! 业务提前终止!!")

        extract_result = poll_response_dict.get('data', {}).get('extract_result', [])[0]
        poll_state = extract_result.get('state')

        if poll_state == "done":
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


@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_download_url: str, local_dir_obj: Path, stem: str) -> Path:
    """下载 MinerU 返回的 zip，解压并定位最终 Markdown 文件。

    解压后的目录结构由 MinerU 决定，Markdown 文件名可能有两种：
    1. 与 PDF 同名（{stem}.md）——直接返回
    2. 统一命名为 full.md ——重命名为 {stem}.md 后返回
    """
    response = requests.get(
        url=zip_download_url,
        timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        logger.error(f"{zip_download_url}地址下载资源报错!业务无法继续进行,提前终止!")
        raise RuntimeError(f"{zip_download_url}地址下载资源报错!业务无法继续进行,提前终止!")
    zip_file_obj: Path = local_dir_obj / f"{stem}.zip"
    zip_file_obj.write_bytes(data=response.content)

    # 清理旧解压目录后重新解压，避免残留文件干扰
    zip_dir_obj: Path = local_dir_obj / stem
    if zip_dir_obj.is_dir():
        shutil.rmtree(zip_dir_obj)
    zip_dir_obj.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_file_obj, zip_dir_obj)

    md_file_list: list[Path] = list(zip_dir_obj.rglob("*.md"))
    if not md_file_list:
        logger.error(f"{zip_download_url}地址下载成功过,解压完成,但是没有md文件!业务无法继续进行,提前终止!")
        raise RuntimeError(f"{zip_download_url}地址下载成功过,解压完成,但是没有md文件!业务无法继续进行,提前终止!")
    # 优先匹配与 PDF 同名的 .md，兜底匹配 full.md
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
    if full_md_file_obj is None:
        raise RuntimeError("解压后的文件中既未找到同名 .md 也未找到 full.md")
    full_md_file_obj.rename(full_md_file_obj.with_stem(f"{stem}.md"))
    return full_md_file_obj


@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """PDF → Markdown 完整编排：校验 → 上传轮询 → 下载解压 → 更新 state。

    此函数是 node_pdf_to_md 节点的核心委托，四个子步骤分别由独立函数承载，
    每步失败均可独立重试。
    """
    pdf_path_obj, local_dir_obj = validate_path_values(state)
    zip_download_url: str = upload_pdf_and_poll(pdf_path_obj)
    md_path_obj: Path = download_and_extract_markdown(zip_download_url, local_dir_obj, pdf_path_obj.stem)
    state['md_path'] = str(md_path_obj)
    return state