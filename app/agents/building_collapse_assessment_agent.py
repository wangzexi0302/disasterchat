import os
import json
import logging
import base64
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Generator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ollama
from shapely import wkt
from collections import OrderedDict
from rasterio.features import rasterize
from app.api.models import AssistantMessage

logger = logging.getLogger(__name__)

class BuildingCollapseAssessmentAgent:
    """
    区域灾害影响范围评估智能体
    """

    def __init__(self, model: str = "qwen2.5"):
        """
        初始化智能体
        """
        # 定义颜色条
        self.color_map = OrderedDict({
            "no-damage": (0, 255, 0, 128),  # 绿色 (B, G, R, alpha)
            "minor-damage": (255, 0, 0, 128),  # 蓝色
            "major-damage": (0, 255, 255, 128),  # 黄色
            "destroyed": (0, 0, 255, 128),  # 红色
            "un-classified": (255, 0, 255, 128)  # 紫色
        })
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

    def _generate_visualization(self, image_post: np.ndarray, polygons: List, damage_levels: List) -> str:
        """
        生成可视化结果
        :param image_post: 灾后影像
        :param polygons: 多边形列表
        :param damage_levels: 损毁等级列表
        :return: 可视化结果路径
        """
        # 创建一个与原图大小相同的透明图层
        overlay = np.zeros((image_post.shape[0], image_post.shape[1], 4), dtype=np.uint8)

        # 将多边形栅格化到图层上
        for polygon, color in zip(polygons, damage_levels):
            mask = rasterize([polygon], out_shape=(image_post.shape[0], image_post.shape[1]), fill=0, default_value=1)
            overlay[mask == 1] = color

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

        return temp_path

    def _generate_statistics(self, damage_counts: Dict) -> Tuple[List, Dict]:
        """
        生成统计结果和柱状图
        :param damage_counts: 损毁统计结果
        :return: 统计结果
        """
        # 定义颜色
        colors = [(r / 255, g / 255, b / 255) for b, g, r, _ in self.color_map.values()]

        # 生成柱状图
        damage_counts_order = OrderedDict()
        temp_files = []
        for i, (surface_feature, damage_type_count) in enumerate(damage_counts.items()):
            dam_type = surface_feature.capitalize()
            dam_type_count_order = OrderedDict()
            for damage_type in self.color_map.keys():
                dam_type_count_order[damage_type] = damage_type_count.get(damage_type, 0)
            damage_counts_order[surface_feature] = dam_type_count_order

            plt.figure(figsize=(8, 6))
            plt.bar(dam_type_count_order.keys(), dam_type_count_order.values(), color=colors)
            plt.xlabel("Damage Level")
            plt.ylabel(f"Number of {dam_type}")
            plt.title(f"{dam_type} Damage Statistics")

            # 创建临时文件保存结果图像
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_path = temp_file.name
                plt.savefig(temp_path, bbox_inches='tight', pad_inches=0.1, dpi=500)
                logger.info(f"已将受灾统计图像保存到临时文件: {temp_path}")
                self.temp_files.append(temp_path) # 记录临时文件路径，以便后续删除

            plt.close()
            temp_files.append(temp_path)
        return temp_files, damage_counts_order

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

    def _post_processing(self,image_post: np.ndarray, label: Dict) -> Tuple[str, List, Dict]:
        """
        对模型输出进行后处理
        :param image_post: 灾后图像
        :param label: 模型输出标签
        """
        # 提取多边形和受损等级
        polygons = []
        damage_levels = []
        damage_counts = {}

        for feature in label["features"]["xy"]:
            feature_type = feature["properties"]["feature_type"]
            if feature_type == "building":
                wkt_data = feature["wkt"]
                subtype = feature["properties"]["subtype"]
                polygon = wkt.loads(wkt_data)  # 使用 shapely 解析 WKT 数据
                polygons.append(polygon)
                damage_levels.append(self.color_map[subtype])

                # 统计建筑物各种程度受损的数量
                damage_counts[feature_type] = damage_counts.get(feature_type, {})
                damage_counts[feature_type][subtype] = damage_counts[feature_type].get(subtype, 0) + 1

        # 生成可视化结果
        image_post_visual_path = self._generate_visualization(image_post, polygons, damage_levels)

        # 生成统计结果和柱状图
        statistics_visual_paths, statistics = self._generate_statistics(damage_counts)

        return image_post_visual_path, statistics_visual_paths, statistics

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
        image_post_visual_path, statistics_visual_paths, statistics = self._post_processing(image_post, label)
        images_content = [self._load_image_as_base64(image_post_visual_path)]
        for path in statistics_visual_paths:
            images_content.append(self._load_image_as_base64(path))

        logger.info(f"运行llm模型{self.model}组织回答...")

        processed_msg = [
            {"role": "system",
            "content": """你是一个专注于灾害影响范围评估的AI助手。我将以JSON格式提供给你建筑的受损情况，请你解读该数据，并组织语言以将该结果反馈给用户。"""},
            {
            "role": 'user',
            "content": f"受损情况统计: {statistics}\n",
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
            message = assistant_message.dict()
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
        image_post_visual_path, statistics_visual_paths, statistics = self._post_processing(image_post, label)
        images_content = [self._load_image_as_base64(image_post_visual_path)]
        for path in statistics_visual_paths:
            images_content.append(self._load_image_as_base64(path))

        logger.info(f"运行llm模型{self.model}组织回答...")

        processed_msg = [
            {"role": "system",
            "content": """你是一个专注于灾害影响范围评估的AI助手。我将以JSON格式提供给你建筑的受损情况，请你解读该数据，并组织语言以将该结果反馈给用户。"""},
            {
            "role": 'user',
            "content": f"受损情况统计: {statistics}\n",
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