from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference

from env_loader import GPT_API_KEY, GPT_BASE_URL, KIMI_BASE_URL, KIMI_API_KEY
from llm_lib import llm
from state import GraphRagState
from tools.tools import search_from_long_term_context, web_search_tool, retrieve_from_milvus


def process_user_input_node(state: GraphRagState, config: RunnableConfig) -> dict:
    """
    处理用户输入，将结果写入state中
    用户输入的是多模态类型, 其中image_url也可以为base64编码的图片
    [
        {"type": "text", "text": "你好"},
        {"type": "image_url", "image_url": "https://example.com/image.jpg"},
    ]
    :param config:
    :param state:
    :return:
    """
    message = state["messages"][-1]
    if not isinstance(message, HumanMessage):
        raise ValueError("Last message must be a HumanMessage")
    content = message.content
    if not isinstance(content, list):
        raise ValueError("Last message must be a list")
    input_text = None
    input_image = None
    for item in content:
        if item["type"] == "text":
            input_text = item["text"]
        elif item["type"] == "image_url":
            input_image = item["image_url"]
    if input_text and input_image:
        input_type = "text_and_image"
    elif input_text and not input_image:
        input_type = "only_text"
    else:
        input_type = "only_image"
    current_user = config["configurable"]["user_id"]
    return {
        "input_type": input_type,
        "input_text": input_text,
        "input_image": input_image,
        "current_user": current_user,
    }


def chat_history_ai_node(state: GraphRagState) -> dict:
    """
    AI节点，使用大模型判断是否在历史聊天记录中存在，如果存在，直接返回
    否则调用工具search_from_milvus，从长期上下文中获取
    :param state:
    :return:
    """
    sys_prompt = """
    你是一个历史消息助手，仅基于用户的历史对话记录回答问题。
    ## 处理流程（按顺序执行）
    1. **优先检查当前会话**
       首先从当前请求携带的会话历史中查找是否有回答用户问题的信息。
       - 若有，直接基于这些信息回答，并注明来自"当前会话"。
    2. **当前会话无答案时调用工具**
       若当前会话历史中无法找到相关信息，调用 `search_from_long_term_context` 工具，从长期历史记录中检索。
    3. **基于工具结果回答**
       - 若工具返回了相关历史记录，结合这些内容回答用户，并注明来自"长期历史记录"。
       - 若工具返回结果为"未找到相关历史记录"或类似表述，请回复：**"空字符串"**
    ## 禁止事项
    - 禁止在历史记录无相关信息时编造或猜测答案。
    - 禁止跳过第一步直接调用长期记忆工具。
    """
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            {"role": "system", "content": sys_prompt},
            MessagesPlaceholder(variable_name="messages", optional=False)
        ]
    )
    chain = prompt | llm.bind_tools([search_from_long_term_context])
    resp = chain.invoke({"messages": state["messages"]})
    return {
        "messages": resp,
    }


def context_ai_node(state: GraphRagState) -> dict:
    """
    根据从milvus中检索到的数据，总结生成答案
    :param state:
    :return:
    """
    sys_prompt = """
    你是一个 Apache Flink 智能助手，专门解答与 Flink 相关的问题。

    ## 处理流程
    
    1. **判断问题归属**
       首先判断用户问题是否与 Flink（Apache Flink）相关。
       - 若**相关**，进入第2步。
       - 若**不相关**，直接回答：**“仅支持Flink相关的问题。”** 无需调用任何工具或进行额外处理。
    
    2. **Flink 相关问题：调用工具检索**
       当问题与 Flink 相关时，必须调用工具 `search_from_milvus` 进行检索。
       - 该工具返回的结果包含**文本**和**图片**两类信息。
       - 在收到工具返回结果之前，请勿提前作答。
       - 收到返回结果后，综合文本内容和图片信息给出完整答复，图片信息需用文字描述其关键内容。
       - 在回答末尾注明信息来源，格式为：`来源：[具体来源名称或链接]`。
       - 若检索结果不足以完整回答问题，返回：**“检索结果不足，无法回答。”** 严禁编造或猜测任何内容。
    
    ## 核心原则
    - 仅处理 Flink 相关问题，非 Flink 问题**直接拒绝回答**。
    - Flink 相关问题**必须**调用 `search_from_milvus`，不得基于自有知识作答。
    - 检索结果不足以回答时，**必须**按指定话术返回，严禁给出任何解释、猜测或编造的内容。
    """
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "{messages}"}
        ]
    )
    chain = prompt | llm.bind_tools([retrieve_from_milvus, web_search_tool])
    return {
        "messages": chain.invoke({"messages": state["messages"]})
    }


async def evaluate_node(state: GraphRagState) -> dict:
    """
    评估节点，根据用户输入、检索到的上下文以及生成的答案，判断答案是否符合要求
    :param state:
    :return:
    """
    answer = state["messages"][-1].content
    sample = SingleTurnSample(
        user_input=state["input_text"],
        retrieved_contexts=state["retrieved_docs"],
        response=answer,
    )
    ragas_llm = llm_factory(
        model="gpt-4o-mini",
        client=AsyncOpenAI(
            api_key=GPT_API_KEY,
            base_url=GPT_BASE_URL,
        )
    )
    precision = LLMContextPrecisionWithoutReference(
        llm=ragas_llm,
    )
    precision_result = await precision.single_turn_ascore(sample)
    return {
        "precision": precision_result,
    }


def human_interrupt_node(state: GraphRagState) -> dict:
    """
    人工介入节点，如果评估结果不合适，需要人工确认时进入此节点
    :param state:
    :return:
    """
    pass


def web_search_ai_node(state: GraphRagState) -> dict:
    """
    网络搜索节点，针对用户的问题进行网络搜索
    只有纯文本才能走到这里，带有图片的只能从向量数据库检索
    :param state: 全局状态
    :return:
    """
    sys_prompt = """
    你是一个专业的网络搜索助手。你的核心工作方式如下：
    1、理解用户的搜索意图，通过调用工具 web_search_tool 获取网络搜索结果。
    2、基于搜索结果提取关键信息，整合成清晰、有条理的答案。
    3、回答需注明信息来源或引用出处，便于用户追溯核实。
    4、若搜索结果不足或无法获取相关信息，如实告知用户，并提供优化搜索的建议（如更换关键词、缩小范围等），不得编造或猜测答案。
    """
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "{input}"}
        ]
    )

    chain = prompt | llm.bind_tools([web_search_tool])
    return {
        "messages": chain.invoke({"input": state["input_text"]})
    }