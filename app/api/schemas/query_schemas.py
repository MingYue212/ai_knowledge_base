"""
查询接口 Pydantic 模型：/query 与 /query/sse 共用的请求/响应结构。
"""
from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    """历史对话单条消息（多轮指代消解的上下文来源）。"""
    role: str = Field(..., description="消息角色：user / assistant")
    content: str = Field(..., description="消息内容")


class QueryRequest(BaseModel):
    """查询请求体：用户问题 + 可选历史对话。"""
    query: str = Field(..., min_length=1, description="用户问题")
    history: list[HistoryMessage] = Field(default_factory=list, description="历史对话（用于指代消解）")


class QueryResponse(BaseModel):
    """查询响应体：最终答案 + 识别商品 + 改写后问题。"""
    answer: str = Field(..., description="最终答案")
    item_names: list[str] = Field(default_factory=list, description="识别到的商品名称")
    rewritten_query: str = Field(default="", description="改写后的查询（含指代消解）")


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
