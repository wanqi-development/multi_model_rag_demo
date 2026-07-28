from enum import StrEnum

from pydantic import BaseModel
from pymilvus import MilvusClient, Function, FunctionType, DataType

from env_loader import MILVUS_PASSWORD, MILVUS_USER
from milvus.db_retriever import MilvusRetriever
from milvus.milvus_client import client


def create_chat_context_collection():
    """
    创建聊天上下文集合
    :return:
    """
    schema = client.create_schema()
    # id user context category context_dense context_sparse
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("context", DataType.VARCHAR, max_length=8192, enable_analyzer=True,
                     analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
    schema.add_field("message_type", DataType.VARCHAR, max_length=32)
    schema.add_field("user", DataType.VARCHAR, max_length=32)
    schema.add_field("timestamp", DataType.INT64, nullable=False)
    schema.add_field("context_dense", DataType.FLOAT_VECTOR, dim=2560)
    schema.add_field("context_sparse", DataType.SPARSE_FLOAT_VECTOR)
    bm25_func = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["context"],
        output_field_names=["context_sparse"],
    )
    schema.add_function(bm25_func)
    index_param = client.prepare_index_params()
    index_param.add_index(field_name="context_dense", index_type="AUTOINDEX", metric_type="IP")
    index_param.add_index("context_sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                          params={
                              "inverted_index_algo": "DAAT_MAXSCORE",
                              # 1.2 ~ 2.0 (1.2) 词频 (TF) 的饱和度: 高频词的贡献越大，词频影响越线性，饱和度增长越慢(通俗：控制一个词出现多少次才算“多”)
                              "bm25_k1": 1.2,
                              # 0.0 ~ 1.0 (0.75) 文档长度归一化的强度： 文档长度的影响越大，对长文档的惩罚越强
                              # （通俗：控制“长篇大论”相对于“言简意赅”的劣势有多大，旨在避免长文档仅仅因为包含更多词汇而在相似度计算中占据不公平的优势。）
                              "bm25_b": 0.75
                          })
    client.create_collection(
        collection_name="t_chat_context",
        schema=schema,
        index_params=index_param
    )
