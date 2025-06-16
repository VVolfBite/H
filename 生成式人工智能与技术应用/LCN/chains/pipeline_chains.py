"""
将 ❶远端 Spark‑LLM 判定 ❷本地 CNN 训练 ❸本地 RNN 训练
封装为 LangChain Tools，供 agent 调用。
"""
from __future__ import annotations
import os, json, importlib, logging
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_core.runnables import Runnable

# ──────────────────────────────────────────────────────────────────────
# 动态载入你 models/ 目录下的脚本
# ──────────────────────────────────────────────────────────────────────
cnn_script = importlib.import_module("models.cnn_mnist")   # 需有 train_model(save_path)
rnn_script = importlib.import_module("models.rnn_stock")   # 需有 train_model(save_path)
spark_llm  = importlib.import_module("models.remote_llm")  # RemoteLLM

LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 1) Spark‑LLM 判断任务类型
# ──────────────────────────────────────────────────────────────────────
_llm = spark_llm.RemoteLLM(
    appid     = "4b69b0da",
    api_key   = "44d76e7af568905bfaf6e7c1b332a98a",
    api_secret= "YmRlZTc0ZTkxNDQ5ZWE2Njk2ZTUwYTFi",
    gpt_url   = "wss://spark-api.xf-yun.com/v1.1/chat"
)

@tool
def classify_task(description: str) -> str:
    """
    使用星火 LLM 判断任务是 **CNN** 还是 **RNN**。\n
    返回: `"CNN"` / `"RNN"` / `"无法识别"`
    """
    result = _llm.classify_task(description)
    LOGGER.info(f"[Spark LLM] {description} → {result}")
    return result


# ──────────────────────────────────────────────────────────────────────
# 2) 训练手写体 CNN
# ──────────────────────────────────────────────────────────────────────
@tool
def train_mnist(_: str = "") -> str:
    """
    训练 MNIST‑CNN。\n
    保存到 `./artifacts/cnn_mnist.pth`，并返回该路径。
    """
    os.makedirs("artifacts", exist_ok=True)
    path = os.path.abspath("artifacts/cnn_mnist.pth")
    cnn_script.train_model(save_path=path)
    return path


# ──────────────────────────────────────────────────────────────────────
# 3) 训练股票 LSTM
# ──────────────────────────────────────────────────────────────────────
@tool
def train_stock(_: str = "") -> str:
    """
    训练股票预测 LSTM。\n
    保存到 `./artifacts/rnn_stock.pth`，并返回该路径。
    """
    os.makedirs("artifacts", exist_ok=True)
    path = os.path.abspath("artifacts/rnn_stock.pth")
    rnn_script.train_model(save_path=path)
    return path


# ──────────────────────────────────────────────────────────────────────
# 4) 自动：先判定 → 再训练
# ──────────────────────────────────────────────────────────────────────
class _AutoTrainRunnable(Runnable):
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        desc = inputs.get("description", "")
        task = _llm.classify_task(desc)

        if task == "CNN":
            path = train_mnist()
        elif task == "RNN":
            path = train_stock()
        else:
            path = None

        return {"task": task, "model_path": path}

@tool
def auto_train(description: str) -> str:
    """根据描述自动判定并训练模型，返回 JSON 结果。"""
    out = _AutoTrainRunnable().invoke({"description": description})
    return json.dumps(out, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────
# 导出一个统一的工具列表，main.py 直接 import 使用
# ──────────────────────────────────────────────────────────────────────
TOOLS = [classify_task, train_mnist, train_stock, auto_train]
