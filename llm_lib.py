import base64
import os

import dashscope
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from env_loader import GPT_BASE_URL, GPT_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, KIMI_API_KEY, \
    KIMI_BASE_URL, LOCAL_BASE_URL, LOCAL_API_KEY

# llm = init_chat_model(
#     model="gpt-4o-mini",
#     api_key=GPT_API_KEY,
#     base_url=GPT_BASE_URL,
#     model_provider="openai",
#     temperature=0.1,
# )

# resp1 = llm.invoke("你好")
# print(resp1)


# llm_bak = init_chat_model(
#     model="gpt-4o-mini",
#     api_key=GPT_API_KEY,
#     base_url=GPT_BASE_URL,
#     model_provider="openai",
#     temperature=0.1,
# )
#
#
# resp1 = llm_bak.invoke("你好")
# print(resp1)

llm = init_chat_model(
    model="kimi-k2.6",
    api_key=KIMI_API_KEY,
    base_url=KIMI_BASE_URL,
    model_provider="openai",
    temperature=1
)


# llm = init_chat_model(
#     model="qwen3-8b",
#     api_key=LOCAL_API_KEY,
#     base_url=LOCAL_BASE_URL,
#     model_provider="openai",
#     temperature=0.1,
#     max_tokens=32768,
# )

def multi_model_embedding(text: str = None, image_path: str = None):
    query = []
    image = ""
    if image_path:
        if image_path.startswith("http"):
            image = image_path
        elif os.path.exists(image_path):
            suffix = os.path.basename(image_path).split(".")[1]
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                base64_content = base64.b64encode(image_bytes).decode("utf-8")
                image = f"data:image/{suffix};base64,{base64_content}"
        else:
            image = ""
    if image:
        query.append({"image": image})
    if text:
        query.append({"text": text})
    resp = dashscope.MultiModalEmbedding.call(
        model="qwen3-vl-embedding",
        api_key=DASHSCOPE_API_KEY,
        input=query,
        enable_fusion=True,
        dimensions=2560
    )
    return resp.output["embeddings"][0]["embedding"]



multi_model_llm = init_chat_model(
    model="qwen3-omni-flash",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    model_provider="openai",
    temperature=0.1,
)


# model_name = "BAAI/bge-large-en-v1.5"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# embedding_model = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
# )



embedding_model = OpenAIEmbeddings(
    api_key=GPT_API_KEY,
    base_url=GPT_BASE_URL,
    model_provider="text-embedding-3-large",
)