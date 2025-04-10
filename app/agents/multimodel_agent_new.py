import os
import logging
from typing import Dict, List, Generator, Optional, Any
import ollama
from app.api.models import AssistantMessage

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

    def _process_image(self, image_path: str) -> str:
        """
        处理图片路径

        Args:
            image_path: 图片文件的路径

        Returns:
            图片文件的路径
        """
        logger.debug(f"开始处理图片路径: {image_path}")
        if not os.path.exists(image_path):
            logger.error(f"图片文件 {image_path} 不存在")
            raise FileNotFoundError(f"图片文件 {image_path} 不存在")
        return image_path

    def _process_messages(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        处理并转换消息格式，提取图像文件

        Args:
            messages: 原始消息列表

        Returns:
            处理后的消息列表和图片文件路径列表
        """
        logger.info("处理多模态消息")
        processed_messages = []
        image_paths = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # 检查是否包含图像
            if isinstance(content, list):
                logger.debug(f"消息 {i + 1} 包含多模态内容")
                text_parts = []

                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        logger.info("处理图像内容")
                        image_path = self._process_image(item.get("image_data", ""))
                        image_paths.append(image_path)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        logger.debug("处理文本内容")
                        text_parts.append(item.get("text", ""))

                # 构造新的消息格式，使用新的API结构
                processed_msg = {
                    "role": role,
                    "content": " ".join(text_parts) if text_parts else "",
                }

                # 如果有图片，添加images字段
                if image_paths:
                    processed_msg["images"] = image_paths

                processed_messages.append(processed_msg)
            else:
                logger.debug(f"消息 {i + 1} 是纯文本内容")
                processed_messages.append({"role": role, "content": content})

        logger.info(f"消息处理完成，共有 {len(image_paths)} 个图片文件")
        return processed_messages, image_paths

    def run(self, message: str, pic_type: str):
        """
        执行多模态推理（非流式）

        """
        logger.info(f"运行多模态推理")

        # image_paths = get_image_path_by_pictype(pic_type) #根据图片类型获取图片

        ollama_message = [{
            'role': 'user',
            'content': message,
            'images': image_paths
        }]

        system_prompt = {
            'role': 'system',
            'content': '多模态模型的Prompt'
        }
        ollama.insert(0,system_prompt)

        try:
            logger.info("调用Ollama多模态模型")
            response = ollama.chat(
                model=self.model,
                messages=ollama_message
            )

            logger.info("成功获取多模态响应")
            
            return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"多模态推理失败: {str(e)}", exc_info=True)
            raise
        finally:
            pass  # 由于不再使用临时文件，无需清理

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

        processed_messages, image_paths = self._process_messages(messages)

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
            pass  # 由于不再使用临时文件，无需清理
