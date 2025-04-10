import ollama
import json
import logging
from typing import List, Dict, Any, Generator
from app.tools import Tool, models
from app.config import Settings
from app.agents.summary_agent import SummaryAgent

logger = logging.getLogger(__name__)

system_message = """你是一个专业的灾害分析意图识别Agent。你的职责是准确理解用户在灾害场景中的问题意图，并调用合适的专业工具进行分析。

# 主要职责
1. 识别用户消息中的具体意图类型
2. 提取关键参数并判断应该调用哪个专业工具
3. 不要试图自己回答专业问题，应交由专业工具处理

# 可识别的意图类型
- 变化检测：比较灾前灾后图像的变化情况
- 道路通畅分析：判断指定路段是否通畅
- 建筑损毁评估：分析建筑物受损程度
- 一般灾情概览：对灾区整体情况进行描述
- 特定区域分析：对用户指定的具体区域进行详细分析

# 工具调用指南
- 当确定用户意图后，必须调用对应的工具函数进行处理
- 提取所有必要参数，确保参数格式正确
- 如果用户提供的信息不足，可以直接回复询问缺失信息，无需调用工具

# 输出格式要求
- 你不需要自己生成回答，只需调用正确工具
- 你的响应将由SummaryAgent进行整合和优化
- 当无法确定意图时，请说明原因而不是猜测

记住，你的核心任务是识别意图并精确调用工具，专业分析结果将由工具函数及SummaryAgent负责生成。
"""

class SentiModelAgent:
    """
    情感分析Agent
    """

    def __init__(self, model: str="qwen2.5"):
        # 使用直接的默认值而不是Settings.default_model
        self.tools = models
        self.model = model
        logger.info(f"SentiModelAgent初始化，加载了{len(self.tools)}个工具")
    
    def _get_function_definitions(self) -> List[Dict[str, Any]]:
        function_defs = [tool.to_function_definition() for tool in self.tools]
        logger.info(f"获取了{len(function_defs)}个函数定义")
        return function_defs
    
    def _get_tool_by_name(self, name:str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                logger.info(f"找到函数：{name}")
                return tool
        logger.info(f"未找到函数：{name}")
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
            "content": (
                "你是一个专注于灾害管理和应急响应的AI助手。并且可以通过识别图片来分析图片中的灾害信息，"
                "提供防灾减灾建议，以及获取灾害信息。如果用户的问题需要查询特定信息，"
                "请使用提供的工具函数来获取信息。不要编造不存在的工具或函数。"
                "请严格按照markdown的格式输出。"
            )
        }

        ollama_messages.insert(0, system_message)
        logger.info("Ollama消息构建完成:", ollama_messages)

        try:
            first_response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                options={
                    "tools": function_defs
                }
            )
            logger.info("成功获取意图识别Agent响应:", first_response)
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
                    logger.info(f"调用Agent：{function_name}")
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








