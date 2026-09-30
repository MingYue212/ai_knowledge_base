"""
查询接口 Pydantic 模型：/query 与 /query/sse 共用的请求/响应结构。
"""
from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    """历史对话单条消息（多轮指代消解的上下文来源）。"""
    role: str = Field(..., description="消息角色：user / assistant")
    content: str = Field(..., description="消息内容")


class QueryRequest(BaseModel):
    """查询请求体：用户问题 + 可选历史对话 + 当前身份。"""
    query: str = Field(..., min_length=1, description="用户问题")
    history: list[HistoryMessage] = Field(default_factory=list, description="历史对话（用于指代消解）")
    user_id: int | None = Field(default=None, description="当前用户 id（缺失时仅可见默认知识组）")


class QueryResponse(BaseModel):
    """查询响应体：最终答案 + 识别商品 + 改写后问题 + 日志索引。"""
    answer: str = Field(..., description="最终答案")
    item_names: list[str] = Field(default_factory=list, description="识别到的商品名称")
    rewritten_query: str = Field(default="", description="改写后的查询（含指代消解）")
    log_id: int | None = Field(default=None, description="查询日志 id（前端反馈按钮用）")
    answer_type: int = Field(default=0, description="回答类型：0正常/1资料缺失拒答/2权限拒答/3澄清追问")


# ---------------- 查询日志 / 知识缺失 ----------------

class FeedbackRequest(BaseModel):
    """问答帮助度反馈请求。"""
    feedback_status: int = Field(..., description="反馈状态：1 有帮助 / 2 无帮助")
    user_id: int | None = Field(default=None, description="当前用户 id（本人校验用，未选身份可缺省）")


class GapItem(BaseModel):
    """知识缺失记录（一条查询日志）。"""
    id: int = Field(..., description="日志 id")
    task_id: str = Field(default="", description="查询任务 id")
    user_id: int | None = Field(default=None, description="提问人 id（未选身份为 null）")
    query: str = Field(..., description="原始问题")
    rewritten_query: str = Field(default="", description="改写后的问题")
    item_names: list[str] = Field(default_factory=list, description="命中的商品名")
    answer: str = Field(default="", description="答案全文")
    answer_type: int = Field(..., description="回答类型：0正常/1资料缺失拒答")
    feedback_status: int = Field(default=0, description="反馈状态：0未反馈/1有帮助/2无帮助")
    gap_handled: bool = Field(..., description="缺失是否已处理")
    created_at: str = Field(default="", description="记录时间")


class GapListResponse(BaseModel):
    """知识缺失分页列表响应。"""
    total: int = Field(..., description="缺失记录总数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页条数")
    items: list[GapItem] = Field(default_factory=list, description="缺失记录列表")


class GapHandleRequest(BaseModel):
    """标记缺失处理状态请求。"""
    handled: bool = Field(..., description="true 标记已处理 / false 撤销")


class GapSummaryResponse(BaseModel):
    """缺失概览计数响应。"""
    refusal_count: int = Field(..., description="资料缺失拒答条数")
    unhelpful_count: int = Field(..., description="无帮助反馈条数")
    pending_count: int = Field(..., description="待处理条数")


if __name__ == '__main__':
    # 模型自检：合法请求可解析，非法请求被拦截
    request = QueryRequest(query="烫金机怎么使用？", history=[{"role": "user", "content": "上一条问题"}])
    print(f"合法请求解析OK: query={request.query}, history={len(request.history)}条")
    print(f"响应模型OK: {QueryResponse(answer='测试答案').model_dump_json()}")
    try:
        QueryRequest(query="")  # min_length=1 应拦截空问题
        print("空问题未被拦截,校验失效!")
    except Exception:
        print("空问题校验拦截OK")
