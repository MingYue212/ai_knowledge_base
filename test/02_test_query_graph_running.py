import json

from app.process.query.agent.main_graph import query_app
from app.process.query.agent.state import create_default_state

# 冒烟测试：只验证查询图可构建与默认状态模板可用
# （完整 invoke 需要 Milvus/LLM/Reranker 等外部服务，属于端到端验证范畴）
print("查询图节点:", sorted(query_app.get_graph().nodes))
state = create_default_state(task_id="task_001", query="烫金机怎么使用？")
print(f"默认状态:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
