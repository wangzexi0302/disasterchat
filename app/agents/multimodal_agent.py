import base64
import os
import tempfile
import logging
from typing import Dict, List, Generator, Optional, Any, Union
import json
import ollama
from app.config import settings
from app.api.models import ImageContent, TextContent, AssistantMessage

logger = logging.getLogger(__name__)

class MultiModalAgent:
    """
    多模态Agent类，使用llava模型处理图像和文本
    支持流式和非流式响应，与AgentService接口保持一致
    """
    
    def __init__(self, model: str = "llava"):
        """
        初始化多模态Agent
        
        Args:
            model: 要使用的模型名称，默认为llava
        """
        self.model = model
        logger.info(f"MultiModalAgent初始化，默认模型: {model}")
    
    def _process_image(self, image_data: str) -> str:
        """
        处理base64编码的图像数据
        
        Args:
            image_data: base64编码的图像数据
            
        Returns:
            临时图像文件的路径
        """
        logger.debug("开始处理base64图像数据")
        
        # 去除可能存在的base64前缀
        if "base64," in image_data:
            logger.debug("从图像数据中移除base64前缀")
            image_data = image_data.split("base64,")[1]
        
        try:
            # 解码base64图像数据
            image_bytes = base64.b64decode(image_data)
            
            # 创建临时文件保存图像数据
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
                logger.info(f"已将图像保存到临时文件: {temp_path}")
                
            return temp_path
        except Exception as e:
            logger.error(f"处理图像数据时出错: {str(e)}", exc_info=True)
            raise
    
    def _process_messages(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        处理并转换消息格式，提取图像文件
        
        Args:
            messages: 原始消息列表
            
        Returns:
            处理后的消息列表和临时文件路径列表
        """
        logger.info("处理多模态消息")
        processed_messages = []
        temp_files = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # 检查是否包含图像
            if isinstance(content, list):
                logger.debug(f"消息 {i+1} 包含多模态内容")
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        logger.info("处理图像内容")
                        image_path = self._process_image(item.get("image_data", ""))
                        new_content.append({"type": "image", "image_path": image_path})
                        temp_files.append(image_path)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        logger.debug("处理文本内容")
                        new_content.append({"type": "text", "text": item.get("text", "")})
                processed_messages.append({"role": role, "content": new_content})
            else:
                logger.debug(f"消息 {i+1} 是纯文本内容")
                processed_messages.append({"role": role, "content": content})
        
        logger.info(f"消息处理完成，共有 {len(temp_files)} 个临时图像文件")    
        return processed_messages, temp_files
    
    def run(self, messages: List[Dict[str, Any]], model: Optional[str] = None) -> Dict[str, Any]:
        """
        执行多模态推理（非流式）
        
        Args:
            messages: 消息列表，可能包含文本和图像
            model: 可选的模型名称，覆盖默认值
            
        Returns:
            与AgentService.run返回格式兼容的响应
        """
        model_name = model or self.model
        logger.info(f"运行多模态推理，使用模型: {model_name}")
        
        processed_messages, temp_files = self._process_messages(messages)
        
        try:
            logger.info("调用Ollama多模态模型")
            response = ollama.chat(
                model=model_name,
                messages=processed_messages
            )
            
            logger.info("成功获取多模态响应")
            
            # 构造与AgentService.run一致的返回格式
            assistant_message = AssistantMessage(
                content=response.get("message", {}).get("content", ""),
                tool_calls=None  # 多模态模型暂不支持工具调用
            )
            
            return {
                "message": assistant_message.dict(),
                "model": model_name
            }
        except Exception as e:
            logger.error(f"多模态推理失败: {str(e)}", exc_info=True)
            raise
        finally:
            # 清理临时文件
            for file_path in temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.debug(f"已删除临时文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除临时文件 {file_path} 失败: {str(e)}")
    
    def run_stream(self, messages: List[Dict[str, Any]], model: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        执行多模态推理（流式）
        
        Args:
            messages: 消息列表，可能包含文本和图像
            model: 可选的模型名称，覆盖默认值
            
        Yields:
            与AgentService.run_stream返回格式兼容的响应块
        """
        model_name = model or self.model
        logger.info(f"流式运行多模态推理，使用模型: {model_name}")
        
        processed_messages, temp_files = self._process_messages(messages)
        
        try:
            logger.info("开始流式调用Ollama多模态模型")
            stream_response = ollama.chat(
                model=model_name,
                messages=processed_messages,
                stream=True
            )
            
            chunk_count = 0
            for chunk in stream_response:
                chunk_count += 1
                content = chunk.get("message", {}).get("content", "")
                # 构造与AgentService.run_stream一致的返回格式
                yield {
                    "message": {
                        "content": content,
                        "tool_calls": None
                    },
                    "model": model_name,
                    "delta": content  # 兼容流式UI更新
                }
            
            logger.info(f"流式生成完成，共 {chunk_count} 个响应块")
        except Exception as e:
            logger.error(f"流式多模态推理失败: {str(e)}", exc_info=True)
        finally:
            # 清理临时文件
            for file_path in temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.debug(f"已删除临时文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除临时文件 {file_path} 失败: {str(e)}")
