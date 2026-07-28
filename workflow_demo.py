import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from node.nodes import web_search_ai_node
from state import GraphRagState
from tools.tools import web_search_tool

def process_input(state: GraphRagState) -> GraphRagState:
    input_message = state["messages"][-1]
    return {
        "input_text": input_message.content,
    }

builder = StateGraph(GraphRagState)

builder.add_node("process_input", process_input)

builder.add_node("web_search_ai_node", web_search_ai_node)

builder.add_node("tools", ToolNode(tools=[web_search_tool]))

builder.add_edge(START, "process_input")

builder.add_edge("process_input", "web_search_ai_node")

builder.add_conditional_edges("web_search_ai_node", tools_condition)

checkpointer = InMemorySaver()

store = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)

config = RunnableConfig(
    configurable={
        "thread_id": uuid.uuid4().hex,
    }
)

chunks = graph.stream({"messages": [
    HumanMessage(content="李自成的人生事迹")
]}, stream_mode="values", config=config)
for chunk in chunks:
    print(chunk)
