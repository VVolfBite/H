import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

# 禁用 HTTP 和 HTTPS 代理（仅限当前 Python 进程）
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "openchat/openchat_3.5",
    "microsoft/phi-2",
    "Qwen/Qwen-1.5-0.5B-Chat"
]

for model_path in MODELS:
    print(f"正在下载: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

print("全部模型已下载完成！")