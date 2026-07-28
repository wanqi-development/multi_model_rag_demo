import json

from pymilvus import AnnSearchRequest, WeightedRanker

from llm_lib import multi_model_embedding
from milvus.milvus_client import client


class MilvusRetriever:
    def __init__(self,
                 collection_name: str,
                 output_fields: list[str],
                 text_field: str,
                 dense_field: str,
                 sparse_field: str):
        self.collection_name = collection_name
        self.output_fields = output_fields
        self.text_field = text_field
        self.dense_field = dense_field
        self.sparse_field = sparse_field

    def dense_search(self, query: list[float], top_k=10, distance_threshold=0.7) -> list[str]:
        resp = client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field=self.dense_field,
            limit=top_k,
            output_fields=self.output_fields,
            search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        )
        return [item["entity"] for item in resp[0] if item.distance >= distance_threshold]

    def sparse_search(self, query: str, top_k=10, distance_threshold=0.7) -> list[str]:
        resp = client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field=self.sparse_field,
            limit=top_k,
            output_fields=self.output_fields,
            search_params={"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}}  # 搜索时要忽略的低重要性词语的比例。,
        )
        return [item["entity"] for item in resp[0] if item.distance >= distance_threshold]

    def hybrid_search(self,
                      query: list[float],
                      original_query: str,
                      top_k=10,
                      dense_weight=1.0,
                      sparse_weight=1.0,
                      distance_threshold=0.9) -> list[dict]:
        dense_search = AnnSearchRequest(
            data=[query],
            anns_field=self.dense_field,
            param={"metric_type": "IP", "params": {"nprobe": top_k}},
            limit=top_k
        )

        sparse_search = AnnSearchRequest(
            data=[original_query],
            anns_field=self.sparse_field,
            param={"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}},
            limit=top_k
        )

        ranker = WeightedRanker(dense_weight, sparse_weight)
        resp = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_search, sparse_search],
            limit=top_k,
            output_fields=self.output_fields,
            ranker=ranker
        )
        return [item["entity"] for item in resp[0] if item.distance >= distance_threshold]


chat_context_retriever = MilvusRetriever(
    collection_name="t_chat_context",
    output_fields=["context", "message_type", "timestamp"],
    text_field="context",
    dense_field="context_dense",
    sparse_field="context_sparse",
)

article_retriever = MilvusRetriever(
    collection_name="t_article",
    output_fields=["id", "text", "title", "source", "type", "image_url"],
    text_field="text",
    dense_field="dense_vector",
    sparse_field="sparse_vector"
)

if __name__ == '__main__':
    embeddings = multi_model_embedding(text="flink中，有界流?")
    # print(json.dumps(article_retriever.sparse_search("flink核心特性?"), indent=4, ensure_ascii=False))
    print(json.dumps(article_retriever.hybrid_search(embeddings, "flink中，有界流?"),  indent=4, ensure_ascii=False))
