import copy
import ollama
import json
import logging
from typing import List, Dict, Any, Generator
from app.tools import Tool, utils
from app.tools.quantity_calculation import QuantityCalculation
from app.tools.path_calculation import PathCalculation
from app.tools.area_calculation import AreaCalculation
from app.config import Settings
from app.vision_models import ChangeDetectionModel, SemanticSegmentationModel
import os


logger = logging.getLogger(__name__)


class ImageAnalysisAgent:
    """
    遥感影像分析Agent
    """

    def __init__(self, model="qwen2.5"):
        self.available_tools = [QuantityCalculation(), PathCalculation(), AreaCalculation()]
        self.model = model
        logger.info(f"ImageAnalysisAgent初始化，加载了{len(self.available_tools)}个工具")

    def _get_function_definitions(self) -> List[Dict[str, Any]]:
        function_defs = [{'type': 'function', 'function': tool.to_function_definition()} for tool in self.available_tools]
        logger.debug(f"获取了{len(function_defs)}个函数定义")

        return function_defs

    def _get_tool_by_name(self, name: str) -> Tool:
        for tool in self.available_tools:
            if tool.name == name:
                logger.debug(f"找到函数：{name}")
                return tool
        logger.debug(f"未找到函数：{name}")

        return None

    def _process_messages(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        处理并转换消息格式，提取图像文件

        Args:
            messages: 原始消息列表

        Returns:
            处理后的消息列表和临时文件路径列表
        """
        logger.info("处理多模态消息...")
        processed_messages = []
        temp_files = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                logger.debug(f"消息 {i + 1} 包含多模态内容")
                text_parts = []
                image_paths = []

                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_path = utils.process_image(item.get("image_data", ""))
                        image_paths.append(image_path)
                        temp_files.append(image_path)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))

                # 构造新的消息格式，使用新的API结构
                processed_msg = {
                    "role": role,
                    "content": " ".join(text_parts) if text_parts else "",
                }

                # # 如果有图片，添加images字段
                # if image_paths:
                #     processed_msg["images"] = image_paths

                processed_messages.append(processed_msg)
            else:
                logger.debug(f"消息 {i + 1} 是纯文本内容")
                processed_messages.append({"role": role, "content": content})

        logger.info(f"消息处理完成，共有 {len(temp_files)} 个临时图像文件")
        return processed_messages, temp_files

    def _task_processing(self, messages: List[Dict[str, str]]):
        # 1. 从消息中获取遥感影像
        processed_messages, image_paths = self._process_messages(messages)

        # assert len(image_paths) == 2, "需要两张影像，一个受灾前影像，一个受灾后影像"

        # pre_image_path = image_paths[0]
        # post_image_path = image_paths[1]

        # demo数据
        pre_image_path = os.path.join('demo_data', 'pre.png')
        post_image_path = os.path.join('demo_data', 'post.png')

        # 2. 调用遥感影像分析模型
        logger.info(f"运行变化检测模型...")
        # image = utils.load_image_as_base64(os.path.join(demo_data_path, 'image.png'))
        # cd = ChangeDetectionModel(model_path='...', device='cuda')
        # cd_result = cd.detect_change(pre_image_data=pre_image, post_image_data=post_image)
        # cd_mask_png = cd_result['change_map']

        # demo数据
        cd_mask_png_path = os.path.join('demo_data', 'change_detection_mask.png')

        logger.info(f"运行语义分割模型...")
        # ss = SemanticSegmentationModel(model_path='...', device='cuda')
        # ss_result = ss.segment(image_data=pre_image)
        # ss_pre_mask_png = ss_result['segmented_image']
        # ss_result = ss.segment(image_data=post_image)
        # ss_post_mask_png = ss_result['segmented_image']

        # demo数据
        ss_pre_mask_png = os.path.join('demo_data', 'pre_segmentation_mask.png')
        ss_post_mask_png = os.path.join('demo_data', 'post_segmentation_mask.png')

        # 3. 对用户意图进行分析
        logger.info("运行意图分析模型，调用遥感图像后处理工具")

        # 用户输入
        user_messages = [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in processed_messages
        ]
        # 参数信息
        parameter_message = {"role": "user",
                             "content": f"""可提供给你的参数包括：{{
                                "灾前遥感影像图": "pre_img_path: {pre_image_path}",            
                                "灾后遥感影像图": "post_img_path: {post_image_path}",
                                "变化检测掩码图": "change_detection_mask_path: {cd_mask_png_path}",
                                "灾前语义分割掩码图": "pre_segmentation_mask_path: {ss_pre_mask_png}",
                                "灾后语义分割掩码图": "post_segmentation_mask_path: {ss_post_mask_png}",
                                "用户提供的起始点坐标": "point_A: [400, 600]",
                                "用户提供的起始点坐标": "point_B: [900, 1000]"}}"""}
        # 系统提示
        system_message = {
            "role": "system",
            "content": "你是一个专注于灾害管理和应急响应的AI助手，可以通过识别图片来分析图片中的灾害信息，并提供防灾减灾建议。"
                       "请你根据用户的需求调用一种或多种工具函数，以帮助用户解决问题。\n"
                       "注意：不要编造不存在的工具或函数，如无工具。\n"
                       "注意：你可以使用工具来回答问题，但如果工具无法匹配，请直接回答，不要强行调用工具。"
        }

        ollama_messages = copy.deepcopy(user_messages)
        ollama_messages.append(parameter_message)
        ollama_messages.insert(0, system_message)

        # 获取调用Agent函数的定义
        tools = self._get_function_definitions()

        try:
            response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                tools=tools,
                # options={"tools": tools}
            )
            logger.info("成功获取模型响应")
        except Exception as e:
            logger.error(f"调用模型失败：{str(e)}", exc_info=True)
            raise

        assistant_message = response["message"]
        tools_response = []
        # tools_response_text = []

        # 4. agent调用
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            tool_calls = assistant_message["tool_calls"]

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_arguments = tool_call["function"]["arguments"]
                tool = self._get_tool_by_name(function_name)
                if tool:
                    logger.debug(f"调用Agent：{function_name}")
                    try:
                        result = tool.execute(**function_arguments)
                        tools_response.append(result)

                        # for item in result['content']:
                        #     if isinstance(item, dict) and item.get("type") == "text":
                        #         tools_response_text.append(item)
                    except Exception as e:
                        result = {
                            "role": "tool",
                            "name": function_name,
                            "content": f"调用Agent失败：{str(e)}"
                        }
                        tools_response.append(result)
                        # tools_response_text.append(result)
                        logger.error(f"调用Agent失败：{str(e)}", exc_info=True)

        ollama_messages = copy.deepcopy(user_messages)
        ollama_messages.insert(0, system_message)
        if tools_response:
            ollama_messages.extend(tools_response)
        else:
            ollama_messages.append({"role": "tool", "name": "无工具调用", "content": "无工具调用"})
        ollama_messages = json.dumps(ollama_messages, ensure_ascii=False)

        ollama_messages = [
            {"role": "system",
            "content": "你是一个专注于灾害管理和应急响应的AI助手。我将提供给你一些历史对话信息，请你组织语言回答用户的提问。"},
            {"role": "user",
             "content": f"历史对话信息：{ollama_messages}\n"
                        f"注意：如果历史消息中有图片路径，你可以根据图片描述生成文字描述，但不需要展示图片地址！\n"
                        f"注意：如果用户问题和分析的答案不一致，以用户问题为准，重新组织回答！"}]

        return ollama_messages, tools_response

    def run(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
            # 1-4. 调用工具处理任务
        ollama_messages, tools_response = self._task_processing(messages)

        # 5. 组织回答：非流式调用
        logger.info("生成最终回答")

        try:
            answer_response = ollama.chat(
                model=self.model,
                messages=ollama_messages
            )
            logger.info("成功获取最终结果")

            # 构建回答信息
            answer_message = {
                "role": 'assistant',
                "content": [
                    {'type': 'text',
                     'text': answer_response['message']['content']}
                ],
                "done": True
            }

            image_result = []
            for res in tools_response:
                for item in res['content']:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_data = utils.load_image_as_base64(item.get("image_data", ""))
                        item['image_data'] = image_data
                        image_result.append(item)
            answer_message['content'].extend(image_result)

            return answer_message
        except Exception as e:
            logger.error(f"结果生成失败：{str(e)}", exc_info=True)
            raise

    def run_stream(self, messages: List[Dict[str, str]]) -> Generator[Dict[str, Any], None, None]:
        # 1-4. 调用工具处理任务
        ollama_messages, tools_response = self._task_processing(messages)

        # 5. 组织回答：流式调用
        logger.info("生成最终回答")

        try:
            logger.info("流式调用Ollama")
            stream_response = ollama.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True
            )

            chunk_count = 0
            for chunk in stream_response:
                chunk_count += 1
                content = chunk.get("message", {}).get("content", "")

                # 构建回答信息
                answer_message = {
                    "role": 'assistant',
                    "content": [
                        {'type': 'text',
                         'text': content}
                    ],
                    "delta": content,  # 兼容流式UI更新
                    "done": False
                }

                yield answer_message

            logger.info(f"流式生成完成，共 {chunk_count} 个响应块")

            image_result = []
            for res in tools_response:
                for item in res['content']:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_data = utils.load_image_as_base64(item.get("image_data", ""))
                        item['image_data'] = image_data
                        image_result.append(item)
            answer_message['content'].extend(image_result)
            answer_message['done'] = True

            # 返回图片
            yield answer_message
        except Exception as e:
            logger.error(f"流式模型推理失败: {str(e)}", exc_info=True)
            raise