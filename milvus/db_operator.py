from llm_lib import multi_model_embedding
from milvus.data_model import ChatContext
from milvus.milvus_client import client


async def write_chat_context(chat_context: ChatContext):
    """
    写入聊天上下文到Milvus数据库
    :param chat_context: 聊天上下文
    """
    context = chat_context.context
    embeddings = multi_model_embedding(text=context)
    chat_context.context_dense = embeddings
    await client.insert(
        collection_name="t_chat_context",
        data=[{
            "context": context,
            "message_type": chat_context.message_type,
            "timestamp": chat_context.timestamp,
            "context_dense": embeddings
        }])
