import os.path

from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
from app.shared.runtime.logger import node_log, PROJECT_ROOT
from app.process.import_.agent.state import ImportGraphState
from app.shared.utils.task_utils import add_running_task, add_done_task


@node_log('node_pdf_to_md')
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 转 Markdown 节点：委托 rag.import_.pdf_parse_service 调用 MinerU 完成解析，
    并将解析后的 md_path 写回 state。
    """
    add_running_task(state['task_id'], 'node_pdf_to_md')
    state = parse_pdf_to_markdown(state)
    add_done_task(state['task_id'], 'node_pdf_to_md')
    return state

if __name__ == '__main__':
    from app.shared.runtime.logger import logger
    from app.process.import_.agent.state import create_default_state

    logger.info("===== 开始 node_pdf_to_md 节点联调测试 =====")


    test_pdf_path = os.path.join(PROJECT_ROOT,'doc','hak180产品安全手册.pdf')
    test_state = create_default_state(
        task_id='test_pdf2md_task_001',
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT,'output'),
    )

    result = node_pdf_to_md(test_state)
    logger.info(f"md_path:{result['md_path']}")
    logger.info(f"md_content长度:{len(result['md_content'])}")
    logger.info("===== 结束 node_pdf_to_md 节点联调测试 =====")
