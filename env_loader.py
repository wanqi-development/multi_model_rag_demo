import os

from dotenv import load_dotenv

load_dotenv(verbose=True)

GPT_API_KEY = os.getenv('GPT_API_KEY')
GPT_BASE_URL = os.getenv('GPT_BASE_URL')

KIMI_API_KEY = os.getenv('KIMI_API_KEY')
KIMI_BASE_URL = os.getenv('KIMI_BASE_URL')

MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_URL = os.getenv("MILVUS_URL")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL")
