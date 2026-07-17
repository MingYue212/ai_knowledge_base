from pathlib import Path
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log


@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """校验 local_file_path 并解析为路由信号与文件元数据。

    三层防御性校验：
    1. 非空 —— 调用方忘了传值
    2. 后缀分支 —— 目前只支持 .md 和 .pdf，其余拒绝
    3. is_file() —— 文件系统存在性兜底（文件可能在上游校验通过后被删除）

    函数同时承担路由职责：设置 is_md_read_enabled / is_pdf_read_enabled，
    供 main_graph.py 的条件边决定下一个节点。
    """
    local_file_path: str = state.get("local_file_path")

    if not local_file_path:
        logger.error("local_file_path的参数为空!业务无法继续进行,提前终止!!")
        raise ValueError("local_file_path的参数为空!业务无法继续进行,提前终止!!")

    # 后缀分支：设置路由标志，决定下一步进入 node_md_img 还是 node_pdf_to_md
    if local_file_path.lower().endswith(".md"):
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True
        state["pdf_path"] = None
        state["is_pdf_read_enabled"] = False
        logger.info(f"local_file_path:{local_file_path},识别为md文件,后续跳转到node_md_img节点!!")
    elif local_file_path.lower().endswith(".pdf"):
        state["md_path"] = None
        state["is_md_read_enabled"] = False
        state["pdf_path"] = local_file_path
        state["is_pdf_read_enabled"] = True
        logger.info(f"local_file_path:{local_file_path},识别为pdf文件,后续跳转到node_pdf_to_md节点!!")
    else:
        logger.error(f"local_file_path:{local_file_path},既不是md又不是pdf!当前项目不支持该文件类型!请检查!!")
        raise ValueError(f"local_file_path:{local_file_path},既不是md又不是pdf!当前项目不支持该文件类型!请检查!!")

    local_file_path_obj: Path = Path(local_file_path)
    if not local_file_path_obj.is_file():
        logger.error(f"local_file_path:{local_file_path}对应的文件不存在或者是文件夹!业务无法继续进行,提前终止!")
        raise ValueError(f"local_file_path:{local_file_path}对应的文件不存在或者是文件夹!业务无法继续进行,提前终止!")

    file_title = local_file_path_obj.stem
    state["file_title"] = file_title
    return state
