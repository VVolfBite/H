import logging
from langchain.agents import initialize_agent, AgentType

from chains.pipeline_chains import TOOLS

# 导入你远程 Spark LLM 相关代码
from models.remote_llm import RemoteLLM
# 确保SparkLLMWrapper也在remote_llm.py中定义，如果不是，请提供其定义位置。
from models.remote_llm import SparkLLMWrapper

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 初始化远程讯飞星火 LLM 客户端
    appid = "4b69b0da"
    api_key = "44d76e7af568905bfaf6e7c1b332a98a"
    api_secret = "YmRlZTc0ZTkxNDQ5ZWE2Njk2ZTUwYTFi"
    gpt_url = "wss://spark-api.xf-yun.com/v1.1/chat"

    remote_llm = RemoteLLM(appid, api_key, api_secret, gpt_url)
    llm = SparkLLMWrapper(remote_llm)

    # 初始化 Agent
    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # ◼︎ Demo 1：仅判定任务类型
    print("\n=== Demo 1 — 判定任务类型 ===")
    res1 = agent.run("请判断：这是一个图像分类任务，使用卷积神经网络。")
    print("Agent 返回：", res1)

    # ◼︎ Demo 2：自动选择并训练
    print("\n=== Demo 2 — 自动选择并训练 ===")
    query = (
        "这是一个时间序列预测任务，"
        "使用过去 30 天股票数据预测明日收盘价。"
        "请自动选择模型并训练，返回模型保存路径。"
    )
    res2 = agent.run(query)
    print("Agent 返回：", res2)
