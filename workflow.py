import asyncio
import os
import time
import uuid
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.constants import START

from milvus.data_model import ChatContext, MessageType

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


from milvus.db_operator import write_chat_context
from node.nodes import process_user_input_node, chat_history_ai_node, context_ai_node, \
    evaluate_node, human_interrupt_node, web_search_ai_node
from router.router import chat_history_router, context_ai_router, need_human_approval_router, process_user_input_router
from state import GraphRagState
from tools.tools import search_from_long_term_context, web_search_tool, retrieve_from_milvus
from utils.file_utils import image_to_base64
from utils.print_utils import pretty_print_messages

builder = StateGraph(GraphRagState)

builder.add_node("process_user_input_node", process_user_input_node)

builder.add_edge(START, "process_user_input_node")

builder.add_node("chat_history_ai_node", chat_history_ai_node)

builder.add_conditional_edges("process_user_input_node", process_user_input_router)

# builder.add_node("chat_history_retrieve_tool_node", CustomizedToolNode(tools=[search_from_long_term_context]))
builder.add_node("chat_history_retrieve_tool_node", ToolNode(tools=[search_from_long_term_context]))

builder.add_edge("chat_history_retrieve_tool_node", "chat_history_ai_node")

builder.add_conditional_edges("chat_history_ai_node", chat_history_router)

builder.add_node("context_ai_node", context_ai_node)

builder.add_node("context_retrieve_tool_node", ToolNode(tools=[retrieve_from_milvus, web_search_tool]))

builder.add_edge("context_retrieve_tool_node", "context_ai_node")

builder.add_conditional_edges("context_ai_node", context_ai_router)

builder.add_node("evaluate_node", evaluate_node)

builder.add_node("human_approve_node", human_interrupt_node)

builder.add_conditional_edges("evaluate_node", need_human_approval_router)

builder.add_node("web_search_ai_node", web_search_ai_node)

builder.add_node("tools", ToolNode(tools=[web_search_tool]))

builder.add_edge("tools", "web_search_ai_node")

builder.add_conditional_edges("web_search_ai_node", tools_condition)

checkpointer = InMemorySaver()

store = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer, store=store, interrupt_before=["human_approve_node"])


current_user = "zzp"
config = RunnableConfig(
    configurable={
        "thread_id": uuid.uuid4().hex,
        "user_id": current_user
    }
)


def update_state(user_input: Literal["agree", "reject"]):
    graph.update_state(
        config=config,
        values={"approve_result": user_input}
    )


async def execute_workflow(user_input: str):
    """
    执行工作流
    1、判断进入当前方法时，是中断后进入还是正常进入
    2、如果是中断输入，那么将用户的输入结果写入state中，然后执行工作流
    3、如果是正常输入，那么直接执行工作流
    4、执行工作流时产生中断，提示中断信息
    5、工作流执行完成，将本次的历史记录写入Milvus中
    :param user_input: 用户输入
    :return: 工作流输出
    """
    state = graph.get_state(config=config)
    if state.next:
        if user_input == "agree":
            update_state(state, "agree")
        else:
            update_state(state, "reject")
        async for chunk in graph.astream(None, config, stream_mode="values"):
            pretty_print_messages(chunk, last_message=True)
    else:
        # 输入对应的文本和图片的分隔符， 文本#图片
        inputs = user_input.split("#")
        input_text = None
        input_image = None
        if len(inputs) == 2:
            input_text = inputs[0]
            input_image = inputs[1]
        elif len(inputs) == 1:
            if inputs[0].startswith("http"):
                input_image = inputs[0]
            elif os.path.isfile(inputs[0]):
                input_image = image_to_base64(inputs[0])
            else:
                input_text = inputs[0]
        content = []
        if input_text:
            content.append({"type": "text", "text": input_text})
        if input_image:
            content.append({"type": "image", "image_url": input_image})
        async for chunk in graph.astream({"messages": [HumanMessage(content=content)]}, config, stream_mode="values"):
            pretty_print_messages(chunk, last_message=True)

    # 执行到这里，两种场景
    # 1、工作流产生中断
    # 2、工作流执行完成
    state = graph.get_state(config=config)
    if state.next:
        return """大模型已结合检索结果生成回答，请确认是否符合预期。
        输入 agree 确认通过，或 reject 拒绝并终止流程。请确认：agree / reject"""
    else:
        ai_message = state["messages"][-1]
        chat_context = ChatContext(
            context=ai_message.content,
            message_type=MessageType.ASSISTANT,
            user = current_user,
            timestamp=time.time()
        )
        asyncio.create_task(write_chat_context(chat_context))
        return ai_message.content


async def main():
    while True:
        user_input = input("请输入问题，文本和图片用#隔开\n")
        if user_input.lower() in ["quit", "exit"]:
            break
        ai_message = await execute_workflow(user_input)
        print("ai_message:", ai_message)

if __name__ == '__main__':
    asyncio.run(main())

