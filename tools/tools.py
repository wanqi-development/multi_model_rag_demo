import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from zai import ZhipuAiClient

from env_loader import ZHIPU_API_KEY
from llm_lib import multi_model_embedding
from milvus.db_retriever import chat_context_retriever, article_retriever


@tool
def web_search_tool(query: Annotated[str, "网络搜索的查询条件"]) -> str:
    """
    网络搜索工具，根据查询返回搜索结果
    :param query:
    :return:
    """
    client = ZhipuAiClient(api_key=ZHIPU_API_KEY)
    try:
        response = client.web_search.web_search(
            search_engine="search_std",
            search_query=query,
            count=5,  # 返回结果的条数，范围1-50，默认10
            search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
        )
        if not response.search_result:
            return "没有搜索到相关结果"
        return "\n\n".join([item["content"] for item in response.search_result])
    except Exception as e:
        return f"网络搜索失败：{e}"


@tool
def search_from_long_term_context(runtime: ToolRuntime) -> str:
    """
    从milvus数据库中搜索，确认长期上下文中是否有数据
    纯文本场景下，可以使用混合搜索
    带有图片场景下，只能使用向量搜索
    :param runtime: 运行时对象
    :return:
    """
    state = runtime.state
    embeddings = multi_model_embedding(text=state["input_text"], image_path=state["input_image"])
    if state["input_type"] == "only_text":
        results = chat_context_retriever.hybrid_search(embeddings, state["input_text"])
    else:
        results = chat_context_retriever.dense_search(embeddings)

    # 这里要增加ragas相关性的评估，如果评估过不去，依旧按照没有搜到来处理 todo
    if not results:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="没有搜索到相关结果",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            },
            goto="context_ai_node"
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="\n\n".join([res["context"] for res in results]),
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )
    return


@tool
def retrieve_from_milvus(runtime: ToolRuntime) -> str:
    """
    从向量知识库milvus中检索你的问题
    :param runtime: 运行时对象
    :return: 检索结果，结果为json格式，包含docs和images两个字段，分别表示检索到的文档和图片url
    """
    state = runtime.state
    embeddings = multi_model_embedding(text=state["input_text"], image_path=state["input_image"])
    input_type = state["input_type"]
    if input_type == "only_text":
        results: list[dict] = article_retriever.hybrid_search(embeddings, state["input_text"])
    else:
        results: list[dict] = article_retriever.dense_search(embeddings)

    docs = []
    images = []
    for res in results:
        doc_type = res["type"]
        if doc_type == "image":
            images.append(res["image_url"])
        elif doc_type == "text":
            docs.append(res["text"])
    if not images and not docs:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="没有检索到相关结果",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            },
            goto = "web_search_ai_node"
        )
    else:
        return Command(
            update={
                "retrieved_docs": docs,
                "retrieved_images": images,
                "messages": [
                    ToolMessage(
                        content=json.dumps({"docs": docs, "images": images}),
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )

#
#
# if __name__ == '__main__':
# results = web_search_tool.invoke("贵阳市今日天气")
# print(json.dumps(results, indent=4, ensure_ascii=False))
# print(results)
