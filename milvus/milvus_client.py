from pymilvus import MilvusClient

from env_loader import MILVUS_PASSWORD, MILVUS_USER,MILVUS_URL , MILVUS_DB_NAME

client = MilvusClient(
    uri=MILVUS_URL,
    user=MILVUS_USER,
    password=MILVUS_PASSWORD,
    db_name=MILVUS_DB_NAME
)
