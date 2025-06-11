from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

class LangChainIntegration:
    def __init__(self):
        load_dotenv()
        
        # 初始化LLM
        self.llm = HuggingFaceHub(
            repo_id="meta-llama/Llama-2-7b-chat-hf",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
        
        # 创建提示模板
        self.analysis_template = PromptTemplate(
            input_variables=["cnn_result", "cnn_accuracy", "rnn_result", "rnn_accuracy"],
            template="""
            分析以下模型预测结果：

            CNN模型结果：
            - 预测类别：{cnn_result}
            - 模型准确率：{cnn_accuracy}%

            RNN模型结果：
            - 预测值：{rnn_result}
            - 模型准确率：{rnn_accuracy}%

            请提供详细分析，包括：
            1. 各模型的表现评估
            2. 可能的改进建议
            3. 整体系统的命中率分析
            """
        )
        
        # 创建记忆组件
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 创建分析链
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_template,
            memory=self.memory,
            verbose=True
        )

    def analyze_results(self, cnn_result, cnn_accuracy, rnn_result, rnn_accuracy):
        """分析模型预测结果并生成报告"""
        analysis = self.analysis_chain.run(
            cnn_result=cnn_result,
            cnn_accuracy=cnn_accuracy,
            rnn_result=rnn_result,
            rnn_accuracy=rnn_accuracy
        )
        return analysis

    def generate_report(self, training_history, test_results):
        """生成完整的实验报告"""
        report_template = PromptTemplate(
            input_variables=["training_history", "test_results"],
            template="""
            基于以下实验数据生成详细报告：

            训练历史：
            {training_history}

            测试结果：
            {test_results}

            请生成包含以下内容的报告：
            1. 实验设置和模型架构
            2. 训练过程分析
            3. 测试结果详细分析
            4. 模型性能评估
            5. 改进建议
            """
        )
        
        report_chain = LLMChain(
            llm=self.llm,
            prompt=report_template,
            verbose=True
        )
        
        report = report_chain.run(
            training_history=training_history,
            test_results=test_results
        )
        return report 