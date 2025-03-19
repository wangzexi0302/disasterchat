import os
import json
import logging
import base64
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Generator
import cv2
import numpy as np
import ollama
from shapely import wkt
from collections import OrderedDict
from rasterio.features import rasterize
from app.api.models import AssistantMessage

logger = logging.getLogger(__name__)

class DisasterImpactAssessmentAgent:
    """
    区域灾害影响范围评估智能体
    """

    def __init__(self, model: str = "qwen2.5"):
        """
        初始化智能体
        """
        self.model = model
        self.temp_files = []
        logger.info("ImpactAssessmentAgent 初始化")

    def _load_label(self, label_path: str) -> Dict[str, Any]:
        """
        加载标签文件
        :param label_path: 标签文件路径
        :return: 标签数据
        """
        if not os.path.exists(label_path):
            logger.error(f"变化检测模型运行失败，请重试！")
            raise FileNotFoundError(f"变化检测模型运行失败，请重试！")

        with open(label_path, "r") as f:
            label = json.load(f)
        return label

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        加载影像
        :param image_path: 影像路径
        :return: 影像数据
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"无法加载影像: {image_path}")
            raise ValueError(f"无法加载影像: {image_path}")
        return image

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
                logger.info(f"已将用户输入图像保存到临时文件: {temp_path}")
                self.temp_files.append(temp_path)  # 记录临时文件路径，以便后续删除
            return temp_path
        except Exception as e:
            logger.error(f"处理图像数据时出错: {str(e)}", exc_info=True)
            raise

    def _load_image_as_base64(self, image_path):
        """将图像文件加载为base64编码字符串"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            logger.error(f"加载图像失败: {str(e)}")
            return ''

    def _generate_visualization(self, image_post: np.ndarray, damage_counts: Dict) -> Tuple[str, Dict]:
        """
        生成可视化结果
        :param image_post: 灾后影像
        :param damage_counts: 多边形列表
        :return: 可视化结果路径
        """
        # 创建一个与原图大小相同的透明图层
        overlay = np.zeros((image_post.shape[0], image_post.shape[1], 4), dtype=np.uint8)

        color = (0, 0, 255, 128)   # 红色，半透明
        area_meters_dict = {}

        # 将多边形栅格化到图层上
        for feature_type, feature_value in damage_counts.items():
            area_meters_sum = 0
            for polygon, area_meters in feature_value:
                mask = rasterize([polygon], out_shape=(image_post.shape[0], image_post.shape[1]), fill=0, default_value=1)
                overlay[mask == 1] = color
                area_meters_sum += area_meters
            area_meters_dict[feature_type] = area_meters_sum

        # 将叠加图层与原图合并
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        image_result = np.uint8(image_post * (1 - alpha) + overlay[:, :, :3] * alpha)

        # 创建临时文件保存结果图像
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            # temp_file.write(image_result)
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image_result)
            logger.info(f"已将受灾标注图像保存到临时文件: {temp_path}")
            self.temp_files.append(temp_path) # 记录临时文件路径，以便后续删除

        return temp_path, area_meters_dict

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
                        logger.info("处理图像内容...")
                        image_path = self._process_image(item.get("image_data", ""))
                        image_paths.append(image_path)
                        temp_files.append(image_path)
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

        logger.info(f"消息处理完成，共有 {len(temp_files)} 个临时图像文件")
        return processed_messages, temp_files

    def _post_processing(self,image_post: np.ndarray, label: Dict) -> Tuple[str, Dict]:
        """
        对视觉模型输出进行后处理
        :param image_post: 灾后图像
        :param label: 模型输出标签
        """
        # 提取多边形和受损等级
        damage_counts = {}
        gsd = label["metadata"]["gsd"]

        for feature in label["features"]["xy"]:
            subtype = feature["properties"]["subtype"]
            if subtype == 'minor-damage' or subtype == 'major-damage' or subtype == 'destroyed':
                feature_type = feature["properties"]["feature_type"]

                # 计算各类地物的受灾面积
                wkt_data = feature["wkt"]
                polygon = wkt.loads(wkt_data)  # 使用 shapely 解析 WKT 数据

                # 转换为实际地理面积（平方米）
                area_meters = polygon.area * (gsd ** 2)

                damage_counts[feature_type]  = damage_counts.get(feature_type, [])
                damage_counts[feature_type].append((polygon, area_meters))

        # 生成可视化结果
        image_post_visual_path, area_meters = self._generate_visualization(image_post, damage_counts)

        return image_post_visual_path, area_meters

    def run(self, messages: List[Dict[str, Any]]):
        """
        执行区域灾害影响范围评估
        :param messages: 消息列表，可能包含文本和图像
        :return: 评估结果
        """
        logger.info("开始执行区域灾害影响范围评估...")

        # 读取影像
        processed_messages, temp_files = self._process_messages(messages)

        assert len(temp_files) == 2, "需要两张影像，一个受灾前影像，一个受灾后影像"

        image_pre = self._load_image(temp_files[0])
        image_post = self._load_image(temp_files[1])

        # 调用变化检测模型
        logger.info(f"运行变化检测模型...")
        # label = change_detection_model(image_pre_path, image_post_path)

        # 加载标签文件
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        label_path = os.path.join(root_path, "./demo_data/hurricane-florence_00000018_post_disaster.json")
        label = self._load_label(label_path)

        # 结果后处理
        logger.info(f"结果后处理...")
        image_post_visual_path, area_meters = self._post_processing(image_post, label)
        images_content = [self._load_image_as_base64(image_post_visual_path)]

        logger.info(f"运行llm模型{self.model}组织回答...")

        processed_msg = [
            {"role": "system",
            "content": """你是一个专注于灾害影响范围评估的AI助手。我将以JSON格式提供给你各类地物（如建筑物）的受灾面积（单位：平方米），请你解读该数据，并组织语言以将该结果反馈给用户。"""},
            {
            "role": 'user',
            "content": f"受灾面积统计: {area_meters}\n",
        }]

        try:
            logger.info("调用Ollama生成回答...")
            response = ollama.chat(
                model=self.model,
                messages=processed_msg
            )

            logger.info("成功获取响应")

            # 构造与AgentService.run一致的返回格式
            assistant_message = AssistantMessage(
                content=response.get("message", {}).get("content", ""),
                tool_calls=None  # 多模态模型暂不支持工具调用
            )
            message = assistant_message.model_dump()
            message["image"] = images_content

            return {
                "message": message,
                "model": self.model,
                "done": True
            }
        except Exception as e:
            logger.error(f"模型失败: {str(e)}", exc_info=True)
        finally:
            # 清理临时文件
            for file_path in self.temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.debug(f"已删除临时文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除临时文件 {file_path} 失败: {str(e)}")

    def run_stream(self, messages: List[Dict[str, Any]]):
        """
        执行区域灾害影响范围评估（流式）
        :param messages: 消息列表，可能包含文本和图像
        :return: 评估结果
        """
        logger.info("开始执行区域灾害影响范围评估...")

        # 读取影像
        processed_messages, temp_files = self._process_messages(messages)

        assert len(temp_files) == 2, "需要两张影像，一个受灾前影像，一个受灾后影像"

        image_pre = self._load_image(temp_files[0])
        image_post = self._load_image(temp_files[1])

        # 调用变化检测模型
        logger.info(f"运行变化检测模型...")
        # label = change_detection_model(image_pre_path, image_post_path)

        # 加载标签文件
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        label_path = os.path.join(root_path, "./demo_data/hurricane-florence_00000018_post_disaster.json")
        label = self._load_label(label_path)

        # 结果后处理
        logger.info(f"结果后处理...")
        image_post_visual_path, area_meters = self._post_processing(image_post, label)
        images_content = [self._load_image_as_base64(image_post_visual_path)]

        logger.info(f"运行llm模型{self.model}组织回答...")

        processed_msg = [
            {"role": "system",
            "content": """你是一个专注于灾害影响范围评估的AI助手。我将以JSON格式提供给你各类地物（如建筑物）的受灾面积（单位：平方米），请你解读该数据，并组织语言以将该结果反馈给用户。"""},
            {
            "role": 'user',
            "content": f"受灾面积统计: {area_meters}\n",
        }]

        try:
            logger.info("流式调用Ollama")
            stream_response = ollama.chat(
                model=self.model,
                messages=processed_msg,
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
                    "model": self.model,
                    "delta": content,  # 兼容流式UI更新
                    "done": False
                }

            logger.info(f"流式生成完成，共 {chunk_count} 个响应块")

            # 返回图片
            yield {
                "message": {
                    "content": "",
                    "tool_calls": None
                },
                "model": self.model,
                "image": images_content,
                "done": True  # 标记流式响应已完成
            }
        except Exception as e:
            logger.error(f"流式模型推理失败: {str(e)}", exc_info=True)
        finally:
            # 清理临时文件
            for file_path in self.temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.debug(f"已删除临时文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除临时文件 {file_path} 失败: {str(e)}")