import os

from dotenv import load_dotenv

load_dotenv(verbose=True)

LOCAL_API_KEY = os.getenv('LOCAL_API_KEY')
LOCAL_BASE_URL = os.getenv('LOCAL_BASE_URL')


GPT_API_KEY = os.getenv('GPT_API_KEY')
GPT_BASE_URL = os.getenv('GPT_BASE_URL')

KIMI_API_KEY = os.getenv('KIMI_API_KEY')
KIMI_BASE_URL = os.getenv('KIMI_BASE_URL')

BAIDU_BAIKE_API_KEY = os.getenv("BAIDU_BAIKE_API_KEY")

MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL")
