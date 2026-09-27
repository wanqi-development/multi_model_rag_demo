from state import GraphRagState


def process_user_input_router(state: GraphRagState) -> str:
    """
    处理用户输入之后的路由
    如果用户输入的内容为纯文本，走到chat_history_ai_node
    否则，走到context_ai_node
    :param state:
    :return:
    """
    if state["input_type"] == "only_text":
        return "chat_history_ai_node"
    else:
        return "context_ai_node"

def chat_history_router(state: GraphRagState) -> str:
    """
    从历史聊天记录上下文的ai节点中，跳转下一个节点的路由
    如果存在工具调用，那么走工具调用节点
    如果不存在工具调用，但是内容不为空，说明找到了结果，直接走end节点
    如果不存在工具调用，但是内容为空，说明需要向量数据库中中检索
    :param state:
    :return:
    """
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "chat_history_retrieve_tool_node"
    elif not ai_message.content:
        return "context_ai_node"
    else:
        return "__end__"


def context_ai_router(state: GraphRagState) -> str:
    """
    从向量数据库检索的ai节点出发，跳转下一个节点的路由
    如果存在工具调用，那么走工具调用节点，走工具调用节点
    如果不存在工具调用，但是内容为仅支持Flink相关的问题。，说明需要走网络搜索节点
    如果不存在工具调用，但是内容为空，检索结果不足，无法回答。同样也需要走网络搜索节点
    否则，检索到了符合要求的答案，此时走evaluator节点, 评估答案的准确定
    :param state:
    :return:
    """
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "context_retrieve_tool_node"
    elif ai_message.content == "仅支持Flink相关的问题":
        return "web_search_ai_node"
    else:
        return "evaluate_node"


def need_human_approval_router(state: GraphRagState) -> str:
    """
    人工审批路由
    如果评估结果不合理，则需要人工审批
    如果合理，则不需要审批，直接结束
    :param state:
    :return:
    """
    precision = state["precision"]
    if precision < 0.8:
        return "human_approve_node"
    else:
        return "__end__"


def need_web_search_router(state: GraphRagState) -> str:
    """
    从人工审批结束
    如果用户输入的为approve，那么直接结束
    如果用户输入的为reject，那么需要网络搜索
    :param state:
    :return:
    """
    if state["approve_result"] == "agree":
        return "__end__"
    elif state["approve_result"] == "reject":
        return "web_search_ai_node"
    else:
        raise ValueError(f"Invalid input: {state['approve_result']}")


