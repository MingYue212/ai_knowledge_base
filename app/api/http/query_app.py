"""
查询服务 FastAPI 应用（端口 settings.query_app_port，默认8001）。

接口清单：
- POST /query      同步问答：invoke 查询图，返回最终答案（阻塞至完成）
- POST /query/sse  流式问答：SSE 逐节点推进度，FINAL 事件带最终答案，CLOSE 关闭

SSE 机制与导入服务一致：
- 后台线程跑 query_app.stream(state, stream_mode="updates")，
  每个节点更新后 task_push_queue(task_id) 推一次进度快照（事件 progress）
- 结束推 FINAL（含答案）/ 异常推 ERROR，最后推 CLOSE 关闭 SSE 流
"""
import threading
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from app.shared.utils.path_util import PROJECT_ROOT

from app.api.schemas.query_schemas import QueryRequest, QueryResponse
from app.process.query.agent.main_graph import query_app
from app.process.query.agent.state import create_default_state
from app.infra.persistence.sqlite_gateway import sqlite_gateway
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import logger
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    set_task_result,
    task_push_queue,
    update_task_status,
)

# FastAPI 应用：查询服务（跨域允许前端调用）
app = FastAPI(title=settings.query_app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_allowed_groups(user_id: int | None) -> list[int]:
    """按身份解析可见知识组；未选身份仅默认知识组，身份失效（已被删）同样回退默认组并告警。"""
    try:
        if user_id:
            return sqlite_gateway.resolve_user_group_ids(int(user_id))
    except ValueError as e:
        logger.warning(f"身份解析失败,回退默认知识组:{str(e)}")
    return [sqlite_gateway.get_default_group_id()]


def _run_query_task(task_id: str, query: str, history: list[dict], allowed_group_ids: list[int]) -> None:
    """后台线程：流式跑查询图，逐节点推进度，结束推 FINAL（带答案）/ERROR，最后推 CLOSE。"""
    try:
        # 1.初始化查询状态并标记任务处理中（allowed_group_ids 供检索时做知识组权限过滤）
        state = create_default_state(task_id=task_id, query=query, history=history,
                                     allowed_group_ids=allowed_group_ids)
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 2.逐节点流式执行：每个节点更新后推送一次进度快照
        final_state: dict = {}
        for update in query_app.stream(state, stream_mode="updates"):
            for _node_name, node_state in update.items():
                if isinstance(node_state, dict):
                    final_state.update(node_state)
            task_push_queue(task_id)

        # 3.查询成功：存答案 + 状态置完成 + 推 FINAL 事件（SSE订阅方拿到最终答案）
        answer = str(final_state.get("answer") or "")
        set_task_result(task_id, "answer", answer)
        update_task_status(task_id, TASK_STATUS_COMPLETED, push_queue=True)
        push_to_session(task_id, SSEEvent.FINAL, {
            "task_id": task_id,
            "answer": answer,
            "item_names": list(final_state.get("item_names") or []),
            "rewritten_query": str(final_state.get("rewritten_query") or ""),
        })
        logger.info(f"查询任务完成:{task_id},答案长度:{len(answer)}")
    except Exception as e:
        # 4.查询失败：状态置失败，推 ERROR 事件（不中断其他任务）
        logger.error(f"查询任务失败:{task_id}:{str(e)}", exc_info=True)
        set_task_result(task_id, "error", str(e))
        update_task_status(task_id, TASK_STATUS_FAILED, push_queue=True)
        push_to_session(task_id, SSEEvent.ERROR, {"task_id": task_id, "error": str(e)})
    finally:
        # 5.推 CLOSE 关闭信号：sse_generator 收到后结束流并清理队列
        push_to_session(task_id, SSEEvent.CLOSE, {})


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """同步问答：invoke 查询图，返回最终答案（含识别商品与改写后问题）。"""
    task_id = uuid4().hex
    history = [item.model_dump() for item in request.history]
    state = create_default_state(task_id=task_id, query=request.query, history=history,
                                 allowed_group_ids=_resolve_allowed_groups(request.user_id))
    final_state = query_app.invoke(state)
    answer = str(final_state.get("answer") or "")
    logger.info(f"同步问答完成:{task_id},答案长度:{len(answer)}")
    return QueryResponse(
        answer=answer,
        item_names=list(final_state.get("item_names") or []),
        rewritten_query=str(final_state.get("rewritten_query") or ""),
    )

@app.post("/query/sse")
def query_sse(request: QueryRequest, http_request: Request) -> StreamingResponse:
    """SSE 流式问答：逐节点推送进度事件，FINAL 事件带最终答案，最后 CLOSE 关闭。"""
    task_id = uuid4().hex
    # 1.队列先建（后台线程才有地方推进度），状态置待处理
    create_sse_queue(task_id)
    update_task_status(task_id, TASK_STATUS_PENDING)
    history = [item.model_dump() for item in request.history]

    # 2.后台线程执行查询图，接口立即返回 SSE 流
    threading.Thread(
        target=_run_query_task,
        args=(task_id, request.query, history, _resolve_allowed_groups(request.user_id)),
        daemon=True,
    ).start()
    return StreamingResponse(sse_generator(task_id, http_request), media_type="text/event-stream")


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

    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
