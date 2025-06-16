# tools/remote_llm_tool.py
from langchain.tools import tool
from models.remote_llm import RemoteLLM

llm = RemoteLLM(
    appid     = "4b69b0da",
    api_key   = "44d76e7af568905bfaf6e7c1b332a98a",
    api_secret= "YmRlZTc0ZTkxNDQ5ZWE2Njk2ZTUwYTFi",
    gpt_url   = "wss://spark-api.xf-yun.com/v1.1/chat"
)

@tool
def classify_task_with_xunfei(description: str) -> str:
    """一句话描述任务 → 'CNN' / 'RNN' / '无法识别'（调用星火大模型）"""
    return llm.classify_task(description)
