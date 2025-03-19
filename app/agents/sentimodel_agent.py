import ollama
import json
import logging
from typing import List, Dict, Any, Generator
from app.tools import Tool, models
from app.config import Settings

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
    
    def run(self, messages: List[Dict[str,str]]) -> Dict[str, Any]:

        logger.info("运行意图分析模型")

        #获取调用Agent函数的定义
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
            "content": (
                "你是一个专注于灾害管理和应急响应的AI助手。并且可以通过识别图片来分析图片中的灾害信息，"
                "提供防灾减灾建议，以及获取灾害信息。如果用户的问题需要查询特定信息，"
                "请使用提供的工具函数来获取信息。不要编造不存在的工具或函数。"
            )
        }

        ollama_messages.insert(0, system_message)

        try:
            response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                options={
                    "tools": function_defs
                }
            )
            logger.info("成功获取模型响应")
        except Exception as e:
            logger.error(f"调用模型失败：{str(e)}", exc_info=True)
            raise

        assistant_message = response["message"]

        #检查是否有agent调用
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            tool_calls = assistant_message["tool_calls"]

            augmented_messages = ollama_messages.copy()
            augmented_messages.append(assistant_message)

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                tool = self._get_tool_by_name(function_name)
                if tool:
                    logger.debug(f"调用Agent：{function_name}")
                    try:
                        result = tool.execute(augmented_messages)
                        augmented_messages.append(
                            {
                                "role": "tool",
                                "name": function_name,
                                "content": result["message"].content
                            }
                        )
                    except Exception as e:
                        logger.error(f"调用Agent失败：{str(e)}", exc_info=True)
            logger.info("生成最终回答")
            try:
                second_response = ollama.chat(
                    model = self.model,
                    messages = augmented_messages
                )
                logger.info("成功获取最终结果")
                return second_response
            except Exception as e:
                logger.error(f"最终结果生成失败：{str(e)}", exc_info=True)
                raise
        else:
            #没有agent调用就调用QA_agent
        




                







