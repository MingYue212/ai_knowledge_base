import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger, step_log


@step_log("validate_get_data")
def validate_get_data(state):
    # 1.获取参数
    md_content: str = state.get("md_content")
    file_title: str = state.get("file_title")
    md_path: str = state.get("md_path")

    # 2.非空、文件存在校验
    if not md_content:
        if (not md_path) or (not Path(md_path).is_file()):
            logger.error("md_content为空，md_path也为空或不是文件，业务终止！")
        md_content = Path(md_path).read_text("utf-8")
        state["md_content"] = md_content
    if not file_title:
        # 文件名为空，给予默认值即可
        file_title = Path(md_path).stem or "default"
        logger.warning(f"file_title为空。给予默认值{file_title}")
        state['file_title'] = file_title
    # 统一换行格式
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    return md_content, file_title, md_path


@step_log("split_document_by_title")
def split_document_by_title(md_content, file_title):
    # 1.声明所需变量
    chunks: list[dict[str, Any]] = []
    current_title: str | None = None
    current_title_lines: list[str] = []
    is_code: bool = False

    # 2.按行切割
    doc_lines: list[str] = md_content.split("\n")

    # 3.定义正则，定位标题行
    title_reg = re.compile(r"^\s*#{1,6}\s.+")

    # 4.开始切割
    for line in doc_lines:
        # 去掉前后空白字符
        line_s = line.strip()
        # 如果是空行，跳过
        if not line_s:
            logger.debug(f"当前行为空行,跳过!")
            continue
        # 如果是代码块开头或结尾
        if line_s.startswith("```") or line_s.startswith("~~~"):
            is_code = not is_code
            logger.debug(f"{'进入代码块~' if is_code else '跳出代码块!'}")
            current_title_lines.append(line_s)
            continue
        # 如果是标题
        if not is_code and title_reg.match(line_s):
            # 结算本赛季
            if current_title and len(current_title_lines) >= 2:
                chunks.append({
                    "title": current_title,
                    "content": "\n".join(current_title_lines),
                    "file_title": file_title
                })
            # 如果是连续标题
            if current_title and len(current_title_lines) == 1:
                current_title_lines = [current_title + "_" + line_s]
                continue
            # 如果是孤儿数据
            if not current_title and len(current_title_lines) > 0:
                current_title_lines.append(line_s)
            else:
                current_title_lines = [line_s]
            current_title = line_s
        else:
            current_title_lines.append(line_s)
    # 最后一个赛季
    if current_title and len(current_title_lines) >= 2:
        chunks.append({
            "title": current_title,
            "content": "\n".join(current_title_lines),
            "file_title": file_title
        })
    logger.info(f"已经根据标题进行切块,现有的块:{len(chunks)}")
    return chunks


@step_log("_split_chunk_content")
def _split_chunk_content(chunk):
    # 1.定义sub_chunks
    sub_chunks = []
    # 2.清除chunk.content里的title
    content = chunk.get("content")
    deal_content = content[len(chunk.get("title")) + 1]
    # 3.获取递归切割器对象
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(chunk.get("title") + "\n"),  # 600 - 前缀的长度
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )
    # 4.开始切割
    for idx, text in enumerate(spliter.split_text(deal_content), start=1):
        sub_chunks.append(
            {
                "parent_title": chunk.get("title"),
                "title": f"{chunk.get('title')}_{idx}",
                "file_title": chunk.get("file_title"),
                "content": chunk.get("title") + "\n" + text,
                "part": idx,
            }
        )
    return sub_chunks


@step_log("_merge_chunks_content")
def _merge_chunks_content(refine_chunks):
    """
    合并短chunk的两个条件：
        1.同一个 parent_title
        2.前一个chunk（base）< 400且合并后<= 1000

    合并方式：把next的content去掉 parent_title前缀后，拼接到 base 后
    """
    merged: list[dict[str, Any]] = []
    base_chunk: dict[str, Any] | None = None

    for next_chunk in refine_chunks:
        if base_chunk is None:
            # 把 base 指针指向第一个next_chunk ，等下一个决策是否合并
            base_chunk = next_chunk
            continue

        base_content = base_chunk.get("content")

        # base 太长了，直接收纳，base 指针后移，指向 next
        if len(base_content) > CHUNK_MIN:
            merged.append(base_chunk)
            base_chunk = next_chunk
            continue

        # --- base ≤ CHUNK_MIN，尝试合并---
        base_parent = base_chunk.get("parent_title")
        next_parent = next_chunk.get("parent_title")
        is_same_parent = base_parent and base_parent == next_parent

        # 不同父标题，不合并，直接收纳base，指针后移，指向 next
        if not is_same_parent:
            merged.append(base_chunk)
            base_chunk = next_chunk
            continue

        # 同一个父标题，计算合并后长度：去掉重复的父标题前缀
        next_content = next_chunk.get("content")
        next_cleared = next_content[len(next_parent) + 1]  # +1 去掉换行符
        merged_len = len(base_content) + len(next_cleared)

        if merged_len > CHUNK_MAX_SIZE:
            # 合并后超长了。不合并
            merged.append(base_chunk)
            base_chunk = next_chunk
            continue
        else:
            # 可以合并
            base_chunk["content"] = base_content + "\n" + next_cleared

    # 最后一个base，直接收纳
    if base_chunk:
        merged.append(base_chunk)

    logger.info(f"完成短 chunk 合并，合并后数量: {len(merged)}")
    return merged


@step_log("refine_split_and_merge_chunks")
def refine_split_and_merge_chunks(chunks):
    # 1.定义refine_chunks
    refine_chunks = []
    # 2.精细切割每一个chunk
    for chunk in chunks:
        # 超长再切
        if len(chunk.get("content")) > CHUNK_SIZE:
            refine_chunks.extend(_split_chunk_content(chunk))
        else:
            refine_chunks.append(chunk)
    logger.info(f"chunk完成精细切割！切割后的数量:{len(refine_chunks)}")
    # 3.合并太短的chunk
    refine_chunks = _merge_chunks_content(refine_chunks)
    return refine_chunks


@step_log("padding_chunks_metadata")
def padding_chunks_metadata(chunks):
    for chunk in chunks:
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk.get("title")
        if "part" not in chunk:
            chunk['part'] = 1


@step_log("backup_chunks_json")
def backup_chunks_json(refine_chunks, md_path):
    # 1.获取备份路径：Path对象
    json_path_obj: Path = Path(md_path).parent / f"{Path(md_path).stem}.json"
    # 2.开始备份
    json_path_obj.write_text(json.dumps(refine_chunks, indent=4, ensure_ascii=False),encoding="utf-8")
    logger.info(f"完成chunks数据的备份,备份位置:{str(json_path_obj)}")


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    # 1.从状态获取所需参数并校验
    md_content, file_title, md_path = validate_get_data(state)
    # 2.根据语义切分
    chunks: list[dict[str, Any]] = split_document_by_title(md_content, file_title)
    # 3.精细切割
    refine_chunks: list[dict[str, Any]] = refine_split_and_merge_chunks(chunks)
    # 4.为切割后的chunks补充parent_title和part
    padding_chunks_metadata(refine_chunks)
    # 5.备份chunks
    backup_chunks_json(refine_chunks, md_path)
    # 6.更新state
    state["chunks"] = refine_chunks
    return state
