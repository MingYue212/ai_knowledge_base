from pathlib import Path
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger

def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    # 1. 读取state中数据local_file_path
    local_file_path:str = state.get("local_file_path") # state['local_file_path'] -> KeyError /
    # 2. local_file_path非空校验 -> 空 -> error日志 + 异常抛出
    if not local_file_path:
        # 日志: 体现错误信息 体现关键参数!!
        logger.error(f"local_file_path的参数为空!业务无法继续进行,提前终止!!")
        # BaseException -> Exception -> ValueError -> 提前关键参数的问题
        raise ValueError(f"local_file_path的参数为空!业务无法继续进行,提前终止!!")
    # 3. 使用字符串函数判断后缀名 .md / .pdf 注意: XXX.MD  / PDF   .md  pdf
    # c://xxx/xx/x/x/x/xx.md | xx.pdf  str 转成 lower()  | endswith()
    if local_file_path.lower().endswith(".md"):
        #4. 如果是md  -> state is_md_read_enabled  md_path
        state['md_path'] =local_file_path
        state['is_md_read_enabled'] = True
        state['pdf_path'] = None
        state['is_pdf_read_enabled'] =False
        logger.info(f"local_file_path:{local_file_path},识别为md文件,后续跳转到node_img_md节点!!state={state}")
    elif local_file_path.lower().endswith(".pdf"):
        #5. 如果是pdf -> is_pdf_read_enabled pdf_path
        state['md_path'] = None
        state['is_md_read_enabled'] = False
        state['pdf_path'] = local_file_path
        state['is_pdf_read_enabled'] = True
        logger.info(f"local_file_path:{local_file_path},识别为pdf文件,后续跳转到node_pdf_to_md节点!!state={state}")
    else:
        # 6. 两者都不是 -> error日志 + 异常抛出
        logger.error(f"local_file_path:{local_file_path},既不是md又不是pdf!当前项目不支持该文件类型!请检查!!")
        raise ValueError(f"local_file_path:{local_file_path},既不是md又不是pdf!当前项目不支持该文件类型!请检查!!")

    # 7. 识别file_tile通过local_file_path [os.path -> pathlib Path]
    # c://xxx/xx/x/x/x/xx.md | xx.pdf  str 转成 lower()  | endswith()
    """
      os.path -> 底层是字符串操作工具  os.path.join("1","2","3")
      Path    -> 面向对象操作的工具    .属性 .函数()   [Path是 os.path的升级版本 Path底层是os.path]

      obj:Path=Path() -> 当前文件对应的Path
      obj:Path=Path(原始地址 local_file_path) -> 指定获取地址文件对应Path
      属性: .stem 获取文件名(没有后缀) .name 文件和后缀  .suffix 后缀   .parent  .parents
      函数: is_file() is_dir() [1.非空 2.是文件或者文件夹] .exists() mk_dir()  
    """
    # 8. 更新state = file_title
    local_file_path_obj: Path = Path(local_file_path)
    if not local_file_path_obj.is_file():
        # 没有 / 或者 有是文件夹
        logger.error(f"local_file_path:{local_file_path}对应的文件不存在或者是文件夹!业务无法继续进行,提前终止!")
        raise ValueError(f"local_file_path:{local_file_path}对应的文件不存在或者是文件夹!业务无法继续进行,提前终止!")
    file_tile = local_file_path_obj.stem
    state['file_title'] = file_tile
    # 9. 返回state
    return state

