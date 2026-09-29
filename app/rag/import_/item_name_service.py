"""
主体名称识别服务：从文档切块中识别商品/主体名称（node_item_name_recognition 节点的业务实现）。

识别策略：
- 文件名 + 文档开头切片是商品信息最密集的区域（封面/概述/参数表）
- 取前 ITEM_NAME_CONTEXT_CHUNK_K 个切片（累计不超过 ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS 字符）
- system 提示词约束只输出商品名（product_recognition_system.prompt），
  user 提示词提供文件名+切片上下文（item_name_recognition.prompt）
- 识别结果写入 state["item_name"]，供 node_import_milvus 建立主体名称索引
"""
from langchain_core.messages import HumanMessage, SystemMessage

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.model.lm_utils import get_llm_client
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log


def build_item_name_context(chunks: list[dict]) -> str:
    """取前K个切片拼接识别上下文，累计超限即截断（防止LLM输入超限）。"""
    # 1.逐片拼接正文，累计超过上限即停（切片按文档顺序，开头信息密度最高）
    parts: list[str] = []
    total_chars = 0
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        if total_chars + len(content) > ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"主体识别上下文已达上限{ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS}字符,截断后续切片")
            break
        parts.append(content)
        total_chars += len(content)
    return "\n\n".join(parts)


@step_log("recognize_item_name")
def recognize_item_name(state: ImportGraphState) -> ImportGraphState:
    """识别文档主体名称并写回 state["item_name"]。

    防御性设计（降级不中断）：
    主体识别是"锦上添花"环节——LLM调用失败或识别为空时，
    item_name保持为空字符串，node_import_milvus会自动跳过主体名称索引，
    不影响文档切块的正常入库。
    """
    # 1.已有值则跳过（幂等：允许上游直接指定主体名）
    item_name: str = (state.get("item_name") or "").strip()
    if item_name:
        logger.info(f"item_name已存在:{item_name},跳过识别")
        return state

    # 2.无切片则无识别上下文，直接跳过（不应阻断导入链路）
    chunks: list[dict] = state.get("chunks") or []
    if not chunks:
        logger.warning("chunks为空,无识别上下文,item_name保持为空!")
        return state

    # 3.渲染提示词：system约束输出格式 + user给文件名与开头切片
    file_title: str = (state.get("file_title") or "").strip()
    context = build_item_name_context(chunks)
    system_prompt = load_prompt("product_recognition_system")
    user_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)

    # 4.调用LLM识别（失败降级为空，不中断导入链路）
    try:
        llm = get_llm_client()
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        # 5.清洗识别结果：只取第一行，去掉引号/句号等包裹噪音
        item_name = str(response.content or "").strip()
        item_name = item_name.split("\n")[0].strip().strip('"“”').rstrip("。. ")
        state["item_name"] = item_name
        if item_name:
            logger.info(f"主体名称识别完成:{item_name}")
        else:
            logger.info("主体名称未能识别,返回为空,导入继续!")
    except Exception as e:
        logger.warning(f"主体名称识别失败,item_name保持为空,导入继续:{str(e)}")
        state["item_name"] = ""
    return state
