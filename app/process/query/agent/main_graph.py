"""
查询流程 LangGraph 状态图定义。
节点链：query_entry → item_name_confirm → search_embedding → search_embedding_hyde → rrf → rerank → answer_output
查询入口为用户问题，链路为线性流程：主路问题检索 + HyDE 假设性文档增强检索
两路召回由 RRF 做排名融合，再交由交叉编码器精排后生成答案
（知识图谱/网络搜索明确不做）。
"""
from langgraph.graph import StateGraph, END

from app.process.query.agent.state import QueryGraphState
from app.process.query.agent.nodes.node_query_entry import node_query_entry
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_answer_output import node_answer_output

query_graph_builder = StateGraph(QueryGraphState)

query_graph_builder.add_node(node_query_entry)  # -> 节点名 == 函数名
query_graph_builder.add_node(node_item_name_confirm)  # -> 节点名 == 函数名
query_graph_builder.add_node(node_search_embedding)  # -> 节点名 == 函数名
query_graph_builder.add_node(node_search_embedding_hyde)  # -> 节点名 == 函数名
query_graph_builder.add_node(node_rrf)  # -> 节点名 == 函数名
query_graph_builder.add_node(node_rerank)  # -> 节点名 == 函数名
query_graph_builder.add_node(node_answer_output)  # -> 节点名 == 函数名

query_graph_builder.set_entry_point("node_query_entry")
query_graph_builder.add_edge("node_query_entry", "node_item_name_confirm")
query_graph_builder.add_edge("node_item_name_confirm", "node_search_embedding")
query_graph_builder.add_edge("node_search_embedding", "node_search_embedding_hyde")
query_graph_builder.add_edge("node_search_embedding_hyde", "node_rrf")
query_graph_builder.add_edge("node_rrf", "node_rerank")
query_graph_builder.add_edge("node_rerank", "node_answer_output")
query_graph_builder.add_edge("node_answer_output", END)

query_app = query_graph_builder.compile()
