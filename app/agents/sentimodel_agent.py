import ollama
import json
import logging
from typing import List, Dict, Any, Generator
from app.tools import Tool, models
from app.config import Settings
from app.agents.summary_agent import SummaryAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的灾害分析意图识别与分解Agent。你的职责是准确理解用户在灾害场景中的问题意图，将复杂问题分解为子问题，并为每个子问题调用最合适的专业工具进行分析，请返回结构化的工具调用信息。

# 问题分解能力
- 识别用户提问中包含的多个独立问题或需求
- 将复杂的多层次问题拆分为可独立回答的子问题
- 确保每个子问题能够被单一工具有效处理
- 对于连续性或依赖性问题，确保按照逻辑顺序处理

# 可用工具及使用场景
1. call_qaagent：回答专业知识和一般问答，当用户询问灾害管理知识、防灾减灾知识、专业术语或一般性问题时使用。
2. call_multimodel：提供基于影像的粗粒度分析，当用户需要灾害概况、灾害类型识别或总体估计时使用。
3. call_image_analysis_agent：提供高精度细粒度分析，仅当用户明确要求精确计算，道路可达性、受灾面积统计、建筑物损伤程度等时使用。

# 工具选择指南
- 默认优先使用简单工具：除非用户明确要求精确分析，否则不要调用call_image_analysis_agent
- 对于需要概述或灾害类型等高层信息的请求，优先使用call_multimodel
- 对于不涉及图像分析的知识性问题，使用call_qaagent
- 任何有关于连通性、点间通常状况、受灾面积、建筑物损伤等问题都可以使用call_image_analysis_agent

# 多agent联合处理流程
- 当识别到多个子问题时，为每个子问题独立选择合适的工具
- 可以针对不同方面的问题同时调用不同工具
- 例如："分析这次洪灾的受灾面积并介绍洪灾防御知识"可分解为:
  1. 调用call_image_analysis_agent分析受灾面积(图像分析)
  2. 调用call_qaagent获取洪灾防御知识(知识问答)
- 保持子问题之间的逻辑关系，确保结果可以被整合

# 输出格式要求
- 当无法确定意图时，请使用call_qaagent处理问题
- 对于复杂问题，可以返回多个工具调用

# 工具调用指南
- 为每个子问题提取所有必要参数，确保参数格式正确
- 必须提供message参数，message参数是用户的具体分析请求，当不确定时可以直接使用用户的提问
- 对于call_multimodel和call_image_analysis_agent，必须确定pic_type参数（pre/post/both），而call_qaagent不需要pic_type参数
- 如果用户提供的信息不足，可以直接回复询问缺失信息，无需调用工具
- 不要使用没有定义的工具函数、不要编造不存在的参数

记住，你的任务是选择最适合用户需求的工具，而不是选择最高级的工具。始终遵循"够用即可"的原则，同时确保所有问题都得到妥善处理。
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
    
    def run(self, messages: List[Dict[str,Any]], pic_type: str = 'pre', sample_index: int = 0, **kwargs):

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
            "content": SYSTEM_PROMPT
        }

        ollama_messages.insert(0, system_message)
        logger.info(f"Ollama消息构建完成:{ollama_messages}")

        try:
            logger.info(f"ollama函数调用定义：{function_defs}")
            first_response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                tools=function_defs,
            )
        except Exception as e:
            logger.error(f"调用意图识别Agent失败：{str(e)}", exc_info=True)
            raise

           
        image_list = []
        first_messages = first_response["message"]

        augmented_messages = ollama_messages.copy()
        augmented_messages.append(first_messages)

        print('first_messages:', first_messages)
     
        # 检查是否有工具调用
        if tool_calls := first_messages.get("tool_calls", None):
            logger.info(f"发现工具调用: {tool_calls}")
            
            for tool_call in tool_calls:
                if fn_call := tool_call.get("function"):
                    fn_name = fn_call["name"]
                    fn_args = fn_call["arguments"]
                    fn_args['sample_index'] = sample_index
                
                    
                    tool = self._get_tool_by_name(fn_name)
                    if tool:
                        logger.info(f"调用工具: {fn_name}, 参数: {fn_args}")
                        try:
                            # 执行工具函数并获取结果
                            result_dict = tool.execute(**fn_args, **kwargs)

                            result = result_dict.get('text', '')
                            if 'images' in result_dict:
                                image_list.extend(result_dict['images'])
                            
                            # 如果结果不是字符串，将其序列化为JSON
                            if not isinstance(result, str):
                                result = json.dumps(result, ensure_ascii=False)
                            
                            # 添加工具响应到消息列表
                            augmented_messages.append({
                                "role": "tool",
                                "name": fn_name,
                                "content": result
                            })
                            logger.info(f"工具 {fn_name} 调用成功")
                        except Exception as e:
                            logger.error(f"工具 {fn_name} 调用失败: {str(e)}", exc_info=True)
                            # 可选：添加错误信息到消息列表
                            augmented_messages.append({
                                "role": "tool",
                                "name": fn_name,
                                "content": f"错误: {str(e)}"
                            })
        
        logger.info(f"增强后的消息列表: {augmented_messages}")

        try:
            augmented_messages = augmented_messages[1:]
            summary_agent = SummaryAgent()
            return summary_agent.run_stream(augmented_messages), image_list
        except Exception as e:
            logger.error(f"最终结果生成失败：{str(e)}", exc_info=True)
            return








