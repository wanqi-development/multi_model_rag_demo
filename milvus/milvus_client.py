from pymilvus import MilvusClient

from env_loader import MILVUS_PASSWORD, MILVUS_USER

client = MilvusClient(
    uri="http://39.108.51.204:19530",
    user=MILVUS_USER,
    password=MILVUS_PASSWORD,
    db_name="default"
)
