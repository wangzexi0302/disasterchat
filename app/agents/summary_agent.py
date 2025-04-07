import logging
from typing import Dict, List, Generator, Optional, Any
import ollama
from app.api.models import AssistantMessage
from app.config import Settings

logger = logging.getLogger(__name__)

class SummaryAgent:
    """
    Summary-Agent，用于对最终结果进行总结推理
    
    """

    def __init__(self, model: str=Settings.default_model):

        logger.info(f"正在初始化Summary-Agent模型：{model}")
        self.model = model
    
    def run(self, messages: List[Dict[str, Any]]):

        logger.info("调用Summary-Agent")
        
        ollama_messages = [
            {
                'role': msg['role'],
                'content': msg['content']
            }
            for msg in messages
        ]

        prompt = {
            'role': 'system',
            'content': '这是Summary-Agent的prompt'
        }

        ollama_messages.insert(prompt, 0)

        try:
            response = ollama.chat(
                model=self.model,
                messages = ollama_messages
            )
            logger.info("成功获取Summary-Agent响应")
            return response.get("message",{})
        except Exception as e:
            logger.error(f"QA-Agent推理失败！: {str(e)}", exc_info=True)
            return 
        

    def run_stream(self, messages: List[Dict[str, Any]]):
        
        logger.info("调用Summary-Agent")
        
        ollama_messages = [
            {
                'role': msg['role'],
                'content': msg['content']
            }
            for msg in messages
        ]

        system_prompt = {
            'role': 'system',
            'content': '这是Summary-Agent的prompt'
        }

        ollama_messages.insert(0, system_prompt)

        try:

            stream_response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True
            )
            logger.info("成功获取Summary-Agent的流式响应")
            for chunk in stream_response:
                yield chunk.get("message",{}).get("content", "")
        except Exception as e:
            logger.error(f"Summary-Agent推理失败！: {str(e)}", exc_info=True)
            return
