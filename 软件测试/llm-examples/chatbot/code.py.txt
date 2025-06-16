import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
import os

# 定义可用模型列表
AVAILABLE_MODELS = {
    "TinyLlama-1.1B-Chat": {
        "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "轻量级聊天模型，1.1B参数"
    },
    "OpenChat-3.5": {
        "path": "openchat/openchat_3.5",
        "description": "基于Mistral的开源聊天模型"
    },
    "Phi-2": {
        "path": "microsoft/phi-2",
        "description": "微软开发的2.7B参数模型"
    },
    "Qwen-1.5-0.5B": {
        "path": "Qwen/Qwen-1.5-0.5B-Chat",
        "description": "阿里通义千问0.5B聊天模型"
    }
}

# 设置页面配置
st.set_page_config(
    page_title="本地聊天机器人",
    layout="wide"
)

# 初始化session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "I am an AI assistant, responsible for answering user questions and strictly following user requirements."}
    ]

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

if "current_model" not in st.session_state:
    st.session_state["current_model"] = "TinyLlama-1.1B-Chat"

if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = "I am an AI assistant, responsible for answering user questions and strictly following user requirements."

# 初始化模型参数
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 256

# 创建保存聊天记录的目录
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def save_chat_history(chat_id, messages):
    """保存聊天记录到文件"""
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_chat_history(chat_id):
    """从文件加载聊天记录"""
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_chat_list():
    """获取所有聊天记录文件列表"""
    chats = []
    for file in os.listdir(CHAT_HISTORY_DIR):
        if file.endswith(".json"):
            chat_id = file[:-5]  # 移除.json后缀
            file_path = os.path.join(CHAT_HISTORY_DIR, file)
            timestamp = os.path.getmtime(file_path)
            chats.append({
                "id": chat_id,
                "timestamp": timestamp,
                "date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            })
    return sorted(chats, key=lambda x: x["timestamp"], reverse=True)

# 标题
st.title("💬 本地模型聊天机器人")

# 侧边栏
with st.sidebar:
    st.header("模型选择")
    # 模型选择下拉框
    selected_model = st.selectbox(
        "选择模型",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state["current_model"]),
        help="选择要使用的模型"
    )
    
    # 显示选中模型的描述
    st.info(AVAILABLE_MODELS[selected_model]["description"])
    
    # 如果模型改变，更新session state
    if selected_model != st.session_state["current_model"]:
        st.session_state["current_model"] = selected_model
        st.rerun()
    
    st.header("系统提示词")
    # 系统提示词输入框
    system_prompt = st.text_area(
        "设置AI助手的角色和行为",
        value=st.session_state["system_prompt"],
        help="设置AI助手的角色和行为方式。例如：'我是一个专业的编程助手，擅长解答技术问题。'",
        height=150
    )
    
    # 应用系统提示词
    if st.button("应用提示词"):
        st.session_state["system_prompt"] = system_prompt
        # 更新当前对话的系统提示词
        if st.session_state["messages"] and st.session_state["messages"][0]["role"] == "assistant":
            st.session_state["messages"][0]["content"] = system_prompt
        else:
            st.session_state["messages"].insert(0, {"role": "assistant", "content": system_prompt})
        st.rerun()
    
    st.header("参数设置")
    
    # 最大生成长度滑块
    st.slider(
        "最大生成长度 (Max Tokens)",
        min_value=64,
        max_value=512,
        value=st.session_state["max_tokens"],
        step=64,
        help="控制生成回复的最大长度。",
        key="max_tokens_slider",
        on_change=lambda: setattr(st.session_state, "max_tokens", st.session_state.max_tokens_slider)
    )
    
    st.header("聊天管理")
    
    # 新建聊天按钮
    if st.button("新建聊天"):
        chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["current_chat"] = chat_id
        st.session_state["messages"] = [
            {"role": "assistant", "content": st.session_state["system_prompt"]}
        ]
        st.rerun()
    
    # 显示聊天历史列表
    st.subheader("历史聊天")
    chats = get_chat_list()
    for chat in chats:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"📝 {chat['date']}", key=f"chat_{chat['id']}"):
                st.session_state["current_chat"] = chat["id"]
                loaded_messages = load_chat_history(chat["id"])
                if loaded_messages:
                    st.session_state["messages"] = loaded_messages
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat['id']}"):
                file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat['id']}.json")
                if os.path.exists(file_path):
                    os.remove(file_path)
                if st.session_state["current_chat"] == chat["id"]:
                    st.session_state["current_chat"] = None
                    st.session_state["messages"] = [
                        {"role": "assistant", "content": st.session_state["system_prompt"]}
                    ]
                st.rerun()

# ==================== 模型加载和推理 ====================
@st.cache_resource
def load_model(model_path):
    """加载模型和tokenizer
    
    Args:
        model_path: 模型路径
    
    Returns:
        tokenizer和model的元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

# 加载模型
try:
    model_path = AVAILABLE_MODELS[st.session_state["current_model"]]["path"]
    tokenizer, model = load_model(model_path)
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# ==================== 聊天界面 ====================
# 展示历史消息
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 构建对话模板
    messages = [{"role": "assistant", "content": st.session_state["system_prompt"]}]
    messages.extend([msg for msg in st.session_state["messages"] if msg["role"] != "assistant"])
    
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # 生成回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=st.session_state["max_tokens"]
            )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            st.write(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})
            
            # 保存聊天记录
            if st.session_state["current_chat"]:
                save_chat_history(st.session_state["current_chat"], st.session_state["messages"])
            else:
                chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["current_chat"] = chat_id
                save_chat_history(chat_id, st.session_state["messages"])

