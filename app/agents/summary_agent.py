import logging
from typing import Dict, List, Generator, Optional, Any
import ollama
from app.api.models import AssistantMessage
from app.config import settings

logger = logging.getLogger(__name__)

class SummaryAgent:
    """
    Summary-Agent，用于对最终结果进行总结推理
    
    """

    # 定义静态系统提示
    SYSTEM_PROMPT = {
        'role': 'system',
        'content': '你是一位遥感灾害领域的总结专家。你的任务是根据用户的提问，综合分析所有其他专家提供的信息，提供一个全面、准确、权威的最终回答。确保回答简洁明了，重点突出，并针对灾害应急、灾情评估、灾害监测等遥感应用场景给出实用的见解。必须使用中文，如果有的agent给出了英文的回答，你需要将其翻译成中文再回答给用户。'
    }

    def __init__(self, model: str="qwen2.5"):
        # 使用直接的默认值而不是Settings.default_model
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

        ollama_messages.insert(0, self.SYSTEM_PROMPT)

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
        
        logger.info(f"调用Summary-Agent:{messages}")
        
        ollama_messages = [
            {
                'role': msg['role'],
                'content': msg['content']
            }
            for msg in messages
        ]

        ollama_messages.insert(0, self.SYSTEM_PROMPT)

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
