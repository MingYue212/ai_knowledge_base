"""
导入服务 FastAPI 应用（端口 settings.import_app_port，默认8000）。

接口清单：
- POST /import                        上传文档（.md/.pdf），后台线程跑导入图，立即返回 task_id
- GET  /import/progress/{task_id}      轮询导入进度（状态 + 已完成/运行中节点，中文展示名）
- GET  /import/progress/stream/{task_id}  SSE 实时进度流（text/event-stream）

进度推送机制：
- 导入图各节点通过 task_utils.add_running_task/add_done_task 更新任务清单（不直接推流）
- 后台线程在 graph.stream 每个节点更新后调用 task_push_queue(task_id)，
  把 {status, done_list, running_list} 快照推进 SSE 队列（事件名 progress）
- 结束推 FINAL 事件，异常推 ERROR 事件，最后推 CLOSE（__close__）关闭 SSE 流
"""
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.api.schemas.import_schemas import (
    ChunkCreateRequest,
    ChunkDocumentListResponse,
    ChunkListResponse,
    ChunkMutationResponse,
    ChunkUpdateRequest,
    ImportResponse,
    ProgressResponse,
)
from app.process.import_.agent.main_graph import import_app
from app.process.import_.agent.state import create_default_state
from app.rag.import_.chunk_manage_service import (
    ChunkNotFoundError,
    add_chunk,
    delete_chunk,
    list_chunk_documents,
    list_chunks,
    update_chunk,
)
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import logger
from app.shared.utils.path_util import PROJECT_ROOT
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    task_push_queue,
    update_task_status,
)

# 支持的导入文件类型：.md 直接读取 / .pdf 走解析
IMPORT_SUPPORT_EXTS = {".md", ".pdf"}
# 上传文件保存目录：项目根 output/uploads/
UPLOAD_DIR = PROJECT_ROOT / "output" / "uploads"

app = FastAPI(title=settings.import_app_name)

# CORS：允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_import_task(task_id: str, local_file_path: str) -> None:
    """后台线程：流式跑导入图，逐节点推进度，结束推 FINAL/ERROR，最后推 CLOSE。"""
    try:
        # 1.初始化导入状态并标记任务处理中
        state = create_default_state(task_id=task_id, local_file_path=local_file_path)
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        logger.info(f"导入任务开始执行:{task_id},文件:{local_file_path}")

        # 2.逐节点流式执行：每个节点更新后推送一次进度快照
        for _update in import_app.stream(state, stream_mode="updates"):
            task_push_queue(task_id)

        # 3.导入成功：状态置完成，推 FINAL 事件（SSE订阅方拿到收尾信号）
        update_task_status(task_id, TASK_STATUS_COMPLETED, push_queue=True)
        push_to_session(task_id, SSEEvent.FINAL, {"task_id": task_id, "status": TASK_STATUS_COMPLETED})
        logger.info(f"导入任务完成:{task_id}")
    except Exception as e:
        # 4.导入失败：状态置失败，推 ERROR 事件（不中断其他任务）
        logger.error(f"导入任务失败:{task_id}:{str(e)}", exc_info=True)
        update_task_status(task_id, TASK_STATUS_FAILED, push_queue=True)
        push_to_session(task_id, SSEEvent.ERROR, {"task_id": task_id, "error": str(e)})
    finally:
        # 5.推 CLOSE 关闭信号：sse_generator 收到后结束流并清理队列
        push_to_session(task_id, SSEEvent.CLOSE, {})


@app.post("/import", response_model=ImportResponse)
async def import_document(file: UploadFile = File(...)) -> ImportResponse:
    """上传文档并启动导入任务（.md/.pdf），返回 task_id 供进度查询/SSE订阅。"""
    # 1.校验文件类型：文件名取 Path(...).name 防路径穿越，后缀白名单校验
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in IMPORT_SUPPORT_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型:{suffix or '未知'},仅支持 {sorted(IMPORT_SUPPORT_EXTS)}",
        )

    # 2.保存文件到 output/uploads/（task_id 前缀防重名覆盖）
    task_id = uuid4().hex
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / f"{task_id}_{filename}"
    save_path.write_bytes(await file.read())
    logger.info(f"导入文件已保存:{save_path}")

    # 3.初始化任务状态与 SSE 队列（队列先建，后台线程才有地方推进度）
    create_sse_queue(task_id)
    update_task_status(task_id, TASK_STATUS_PENDING)

    # 4.后台线程执行导入图，接口立即返回 task_id
    threading.Thread(target=_run_import_task, args=(task_id, str(save_path)), daemon=True).start()
    return ImportResponse(task_id=task_id)


@app.get("/import/progress/{task_id}", response_model=ProgressResponse)
def import_progress(task_id: str) -> ProgressResponse:
    """轮询导入进度：状态 + 已完成节点 + 运行中节点（中文展示名）。"""
    return ProgressResponse(
        status=get_task_status(task_id) or TASK_STATUS_PENDING,
        done_list=get_done_task_list(task_id),
        running_list=get_running_task_list(task_id),
    )


@app.get("/import/progress/stream/{task_id}")
def import_progress_stream(task_id: str, request: Request) -> StreamingResponse:
    """SSE 实时进度流：逐节点推送 progress 事件，结束推 final/error 后自动关闭。"""
    return StreamingResponse(sse_generator(task_id, request), media_type="text/event-stream")


# ---------------- 切片管理（手动管理已入库 chunk） ----------------

def _map_chunk_error(e: Exception) -> HTTPException:
    """服务层业务异常 → HTTP 语义：切片不存在 404，其余入参问题 400。"""
    if isinstance(e, ChunkNotFoundError):
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=400, detail=str(e))


@app.get("/chunks/documents", response_model=ChunkDocumentListResponse)
def chunk_documents() -> ChunkDocumentListResponse:
    """列出所有已入库文档及各自的切片数量。"""
    try:
        docs = list_chunk_documents()
    except Exception as e:
        raise _map_chunk_error(e) from e
    return ChunkDocumentListResponse(documents=docs)


@app.get("/chunks", response_model=ChunkListResponse)
def chunk_list(
    file_title: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> ChunkListResponse:
    """按文档标题分页列出切片（按父标题/序号稳定排序）。"""
    try:
        result = list_chunks(file_title=file_title, page=page, page_size=page_size)
    except Exception as e:
        raise _map_chunk_error(e) from e
    return ChunkListResponse(**result)


@app.post("/chunks", response_model=ChunkMutationResponse)
def chunk_create(request: ChunkCreateRequest) -> ChunkMutationResponse:
    """手动新增切片：校验 + 自动向量化 + 入库。"""
    try:
        result = add_chunk(request.file_title, request.parent_title, request.part, request.chunk_text)
    except Exception as e:
        raise _map_chunk_error(e) from e
    return ChunkMutationResponse(new_id=result.get("id"), chunk=result)


@app.patch("/chunks/{chunk_id}", response_model=ChunkMutationResponse)
def chunk_update(chunk_id: int, request: ChunkUpdateRequest) -> ChunkMutationResponse:
    """编辑切片正文/标题：重嵌 → 删旧插新（id 会变化）。"""
    try:
        result = update_chunk(chunk_id, request.chunk_text, request.parent_title, request.part)
    except Exception as e:
        raise _map_chunk_error(e) from e
    return ChunkMutationResponse(
        old_id=result.get("old_id"), new_id=result.get("new_id"), chunk=result["chunk"]
    )


@app.delete("/chunks/{chunk_id}")
def chunk_delete(chunk_id: int) -> dict:
    """删除单个切片。"""
    try:
        delete_chunk(chunk_id)
    except Exception as e:
        raise _map_chunk_error(e) from e
    return {"ok": True, "id": chunk_id}


# 前端控制台（单页应用，两服务共用同一页面，/docs 仍可用作接口调试）
HTML_PATH = PROJECT_ROOT / "app" / "api" / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """返回前端控制台页面。"""
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="前端页面缺失:app/api/static/index.html")
    return FileResponse(HTML_PATH, media_type="text/html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
