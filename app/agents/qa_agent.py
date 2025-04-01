import os
import logging
from typing import Dict, List, Generator, Optional, Any
import ollama
from app.api.models import AssistantMessage
from app.config import Settings

logger = logging.getLogger(__name__)

class QAAgent:
    """
    QA Agent，用于进行地灾知识的专家回答
    
    """

    def __init__(self, model: str=Settings.default_model):

        logger.info(f"正在初始化QA-Agent模型：{model}")
        self.model = model
    
    def run(self, message: str, pic_type: str):
        
        logger.info("运行QA-Agent")
        
        #定义QA-Agent的prompt
        system_prompt = {
            'role': 'system',
            'content': 'QA-Agent的Prompt'
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
            return



