# 本地聊天机器人

这是一个基于DialoGPT-medium的本地聊天机器人，使用Streamlit构建用户界面。

## 功能特点

- 完全本地运行，不需要API密钥
- 基于DialoGPT-medium模型
- 简洁的Streamlit界面
- 支持对话历史记录
- 可清除对话历史

## 安装步骤

1. 克隆仓库
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行方法

```bash
streamlit run local_chatbot.py
```

## 使用说明

1. 启动应用后，在输入框中输入您的问题
2. 等待模型生成回复
3. 可以随时使用"清除对话"按钮重置对话历史

## 注意事项

- 首次运行时会下载模型，可能需要一些时间
- 需要确保有足够的磁盘空间存储模型
- 建议使用GPU运行以获得更好的性能 