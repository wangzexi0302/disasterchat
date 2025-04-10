import os
import logging
from typing import Dict, List, Generator, Optional, Any
import ollama
from app.api.models import AssistantMessage
from app.config import settings

logger = logging.getLogger(__name__)

class QAAgent:
    """
    QA Agent，用于进行地灾知识的专家回答和普通问题解答
    
    """

    def __init__(self, model: str = settings.default_model):

        logger.info(f"正在初始化QA-Agent模型：{model}")
        self.model = model
    
    def run(self, message: str):
        """
        执行QA推理，回答用户的普通问题或专业问题
        
        Args:
            message: 用户输入的问题或咨询内容
            
        Returns:
            模型回答结果
        """
        logger.info("运行QA-Agent")
        logger.info(f"QA-Agent处理问题: {message[:50]}...")
        
        # 定义QA-Agent的prompt
        system_prompt = {
            'role': 'system',
            'content': '你是一个专注于灾害管理和应急响应的AI助手。你可以回答用户关于地质灾害、自然灾害的专业问题，也可以回答一般性问题。请提供专业、准确且有帮助的回答。'
        }

        ollama_message = [system_prompt]
        ollama_message.append(
            {
                'role': 'user',
                'content': message
            }
        )

        try:
            response = ollama.chat(
                model = self.model,
                messages = ollama_message
            )
            logger.info("成功获取QA-Agent响应")
            return response.get("message",{}).get("content","")
        except Exception as e:
            logger.error(f"QA-Agent推理失败！: {str(e)}", exc_info=True)
            return f"问题回答失败: {str(e)}"



