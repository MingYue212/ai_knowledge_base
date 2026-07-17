from langgraph.graph import StateGraph, END

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus

import_graph_builder = StateGraph(ImportGraphState)

import_graph_builder.add_node(node_entry)
import_graph_builder.add_node(node_pdf_to_md)  # -> 节点名 == 函数名
import_graph_builder.add_node(node_md_img)  # -> 节点名 == 函数名
import_graph_builder.add_node(node_document_split)  # -> 节点名 == 函数名
import_graph_builder.add_node(node_item_name_recognition)  # -> 节点名 == 函数名
import_graph_builder.add_node(node_bge_embedding)  # -> 节点名 == 函数名
import_graph_builder.add_node(node_import_milvus)  # -> 节点名 == 函数名

import_graph_builder.set_entry_point("node_entry")


def after_node_entry(state: ImportGraphState):
    if state["is_md_read_enabled"]:
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    else:
        return END


import_graph_builder.add_conditional_edges("node_entry",
                                           after_node_entry,
                                           {
                                               "node_md_img": "node_md_img",
                                               "node_pdf_to_md": "node_pdf_to_md",
                                               END: END
                                           })
import_graph_builder.add_edge("node_pdf_to_md", "node_md_img")
import_graph_builder.add_edge("node_md_img", "node_document_split")
import_graph_builder.add_edge("node_document_split", "node_item_name_recognition")
import_graph_builder.add_edge("node_item_name_recognition", "node_bge_embedding")
import_graph_builder.add_edge("node_bge_embedding", "node_import_milvus")
import_graph_builder.add_edge("node_import_milvus", END)

import_app = import_graph_builder.compile()



