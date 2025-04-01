import ollama
import json
import logging
from typing import List, Dict, Any, Generator
from app.tools import Tool, models
from app.config import Settings
from app.agents.summary_agent import SummaryAgent

logger = logging.getLogger(__name__)

class SentiModelAgent:
    """
    情感分析Agent
    """

    def __init__(self, model: str=Settings.default_model):
        self.tools = models
        self.model = model
        logger.info(f"SentiModelAgent初始化，加载了{len(self.tools)}个工具")
    
    def _get_function_definitions(self) -> List[Dict[str, Any]]:
        function_defs = [tool.to_function_definition() for tool in self.tools]
        logger.debug(f"获取了{len(function_defs)}个函数定义")
        return function_defs
    
    def _get_tool_by_name(self, name:str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                logger.debug(f"找到函数：{name}")
                return tool
        logger.debug(f"未找到函数：{name}")
        return None
    
    def run(self, messages: List[Dict[str,Any]], pic_type: str):

        logger.info("运行意图分析模型")

        #获取可用Agent的定义
        function_defs = self._get_function_definitions()

        #构建Ollama参数，但是判断意图是根据最后一段用户输入还是包括历史输入
        ollama_messages = [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in messages
        ]

        #添加系统提示
        system_message = {
            "role": "system",
            "content": "意图识别Agent的Prompt"
        }

        ollama_messages.insert(0, system_message)

        try:
            first_response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                options={
                    "tools": function_defs
                }
            )
            logger.info("成功获取意图识别Agent响应")
        except Exception as e:
            logger.error(f"调用意图识别Agent失败：{str(e)}", exc_info=True)
            raise

        first_messages = first_response["message"]

        augmented_messages = ollama_messages.copy()
        augmented_messages.append(first_messages)

        #检查是否有agent调用
        if "tool_calls" in first_messages and first_messages["tool_calls"]:
            tool_calls = first_messages["tool_calls"]
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                tool = self._get_tool_by_name(function_name)
                params = tool_call["function"]["arguments"]
                params["pic_type"] = pic_type
                if tool:
                    logger.debug(f"调用Agent：{function_name}")
                    try:
                        result = tool.execute(**params)
                        augmented_messages.append(
                            {
                                'role': 'tool',
                                'name': function_name,
                                'content': result
                            }
                        )
                    except Exception as e:
                        logger.error(f"调用Agent失败：{str(e)}", exc_info=True)
        try:
            augmented_messages = augmented_messages[1:]
            summary_agent = SummaryAgent()
            return summary_agent.run_stream(augmented_messages)
        except Exception as e:
            logger.error(f"最终结果生成失败：{str(e)}", exc_info=True)
            return








