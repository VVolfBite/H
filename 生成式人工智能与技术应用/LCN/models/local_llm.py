from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)

class LocalLLM(LLM, BaseModel):
    
    model_name: str = Field(default="EleutherAI/gpt-neo-1.3B", description="模型名称")
    device: str = Field(default="cuda", description="运行设备")
    temperature: float = Field(default=0.7, description="生成温度")
    tokenizer: Any = Field(default=None, description="分词器")
    model: Any = Field(default=None, description="模型")
    
    def __init__(self, **kwargs):
        """初始化"""
        super().__init__(**kwargs)
        try:
            # 加载分词器和生成模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置 pad_token 为 eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto"
            ).to(self.device)
            logger.info(f"成功加载模型: {self.model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def _call(self, input_text: str) -> str:
        prompt = (
            "You are a classifier. Only respond with either 'CNN' or 'RNN'.\n"
            "Task: Recognize handwritten digits.\nAnswer: CNN\n"
            "Task: Identify objects in images.\nAnswer: CNN\n"
            "Task: Detect cats in pictures.\nAnswer: CNN\n"
            "Task: Forecast next month's sales from past data.\nAnswer: RNN\n"
            "Task: Predict future stock prices.\nAnswer: RNN\n"
            "Task: Model user behavior over time.\nAnswer: RNN\n"
            f"Task: {input_text}\nAnswer:"
        )
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 🔥 只取生成的新内容部分（去掉原始 prompt 部分）
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 清洗 & 提取首词
        result = result.upper().strip().strip(".! ").split()[0]

        return result
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "local_llm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型参数"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "temperature": self.temperature
        } 