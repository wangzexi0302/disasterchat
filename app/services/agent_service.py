import ollama
import json
from typing import List, Dict, Any, Tuple, Generator
from app.tools import available_tools, Tool

class AgentService:
    """LLM Agent服务，支持函数调用"""
    
    def __init__(self):
        self.tools = available_tools
        
    def _get_function_definitions(self) -> List[Dict[str, Any]]:
        """获取所有工具的函数定义"""
        return [tool.to_function_definition() for tool in self.tools]
    
    def _get_tool_by_name(self, name: str) -> Tool:
        """根据名称获取工具实例"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
        
    def run(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """
        运行Agent流程
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            model: 使用的模型名称
            
        Returns:
            Dict包含响应内容
        """
        # 获取所有可用工具的定义
        function_definitions = self._get_function_definitions()
        
        # 构建Ollama请求参数
        ollama_messages = [
            {
                "role": msg["role"],
                "content": msg["content"]
            } for msg in messages
        ]
        
        # 添加系统提示，指导模型作为Agent行为
        system_message = {
            "role": "system", 
            "content": (
                "你是一个专注于灾害管理和应急响应的AI助手。你可以帮助用户了解自然灾害知识，"
                "提供防灾减灾建议，以及获取灾害信息。如果用户的问题需要查询特定信息，"
                "请使用提供的工具函数来获取信息。不要编造不存在的工具或函数。"
            )
        }
        
        # 将系统提示插入到消息列表的开头
        ollama_messages.insert(0, system_message)
        
        # 调用模型
        response = ollama.chat(
            model=model,
            messages=ollama_messages,
            options={
                "tools": function_definitions  # 传递工具定义
            }
        )
        
        # 处理模型响应
        assistant_message = response["message"]
        
        # 检查是否有工具调用
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            # 处理工具调用
            augmented_messages = ollama_messages.copy()
            augmented_messages.append(assistant_message)
            
            # 执行每个工具调用
            for tool_call in assistant_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                # 获取工具并执行
                tool = self._get_tool_by_name(function_name)
                if tool:
                    result = tool.execute(**function_args)
                    
                    # 将工具结果添加到消息中
                    augmented_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": str(result)
                    })
            
            # 使用更新后的消息再次调用模型
            second_response = ollama.chat(
                model=model,
                messages=augmented_messages
            )
            
            # 返回最终结果
            return second_response
        
        # 如果没有工具调用，直接返回原始响应
        return response
    
    def run_stream(self, messages: List[Dict[str, str]], model: str) -> Generator[Dict[str, Any], None, None]:
        """
        流式运行Agent流程
        
        Args:
            messages: 消息列表
            model: 使用的模型名称
            
        Yields:
            Dict: 包含流式响应内容
        """
        # 与非流式版本类似，但返回流式响应
        function_definitions = self._get_function_definitions()
        
        ollama_messages = [
            {
                "role": msg["role"],
                "content": msg["content"]
            } for msg in messages
        ]
        
        system_message = {
            "role": "system", 
            "content": (
                "你是一个专注于灾害管理和应急响应的AI助手。你可以帮助用户了解自然灾害知识，"
                "提供防灾减灾建议，以及获取灾害信息。如果用户的问题需要查询特定信息，"
                "请使用提供的工具函数来获取信息。不要编造不存在的工具或函数。"
            )
        }
        
        ollama_messages.insert(0, system_message)
        
        # 第一阶段：获取模型的初始响应
        complete_response = {"message": {"content": "", "tool_calls": []}}
        
        for chunk in ollama.chat(
            model=model,
            messages=ollama_messages,
            options={"tools": function_definitions},
            stream=True
        ):
            # 累积完整响应
            if "message" in chunk:
                if "content" in chunk["message"]:
                    complete_response["message"]["content"] += chunk["message"]["content"]
                if "tool_calls" in chunk["message"]:
                    complete_response["message"]["tool_calls"] = chunk["message"].get("tool_calls", [])
            
            # 传递流式块
            yield chunk
            
        # 检查是否需要工具调用
        if "tool_calls" in complete_response["message"] and complete_response["message"]["tool_calls"]:
            # 处理工具调用
            augmented_messages = ollama_messages.copy()
            augmented_messages.append(complete_response["message"])
            
            # 执行每个工具调用
            for tool_call in complete_response["message"]["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                # 获取工具并执行
                tool = self._get_tool_by_name(function_name)
                if tool:
                    result = tool.execute(**function_args)
                    
                    # 将工具结果添加到消息中
                    augmented_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": str(result)
                    })
            
            # 使用更新后的消息再次调用模型，并传递流式响应
            yield {"message": {"content": "\n\n基于工具调用的结果，我正在生成最终回答...\n\n"}}
            
            for chunk in ollama.chat(model=model, messages=augmented_messages, stream=True):
                yield chunk
