# -*- coding: utf-8 -*-
import tempfile
from collections import OrderedDict
from app.tools.base import Tool
from typing import Dict, Any
import os
import cv2
import numpy as np
from app.tools.utils import load_image, save_image, print_mask_color
import matplotlib.pyplot as plt


class QuantityCalculation(Tool):
    """数量计算工具"""

    def __init__(self):
        # 定义颜色对应的损伤类别
        self.color_map = OrderedDict({
            "no-damage": (0, 255, 0),  # 无损伤
            "minor-damage": (0, 255, 255),  # 轻度损伤
            "major-damage": (0, 0, 255),  # 重大损伤
            "un-classified": (255, 0, 255),  # 未分类
        })

    @property
    def name(self) -> str:
        return "quantity_calculation"

    @property
    def description(self) -> str:
        return "对灾后建筑物不同受损程度进行统计"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "post_img_path": {
                "type": "string",
                "description": "灾后遥感影像图路径",
                "required": True
            },
            "change_detection_mask_path": {
                "type": "string",
                "description": "受灾前后变化检测的掩码图路径",
                "required": True
            }
        }

    def _visualize_damage_statistics(self, damage_counts: Dict) -> str:
        """
        可视化建筑物损伤统计结果

        :param damage_counts: 建筑物损伤统计
        :return: 可视化结果路径
        """
        colors = [(r / 255, g / 255, b / 255) for b, g, r in self.color_map.values()]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(damage_counts.keys(), damage_counts.values(), color=colors)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

        plt.xlabel("Damage Level")
        plt.ylabel(f"Number of buildings")
        plt.title(f"Damage Statistics")

        # 创建临时文件保存结果图像
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            plt.savefig(temp_path, bbox_inches='tight', pad_inches=0.1, dpi=500)
        # print(f"可视化已保存至: {temp_path}")
        plt.close()

        return temp_path

    def _visualize_damage(self, image_post: np.ndarray, mask_img: np.ndarray) -> str:
        """
        为灾后遥感影像生成可视化结果

        :param image_post: 灾后影像
        :param mask_img: 掩码影像
        :return: 可视化结果路径
        """
        damage_overlay_colors = OrderedDict({value: value + (128,) for key, value in self.color_map.items()})

        # 创建一个与原图大小相同的透明图层
        overlay = np.zeros((image_post.shape[0], image_post.shape[1], 4), dtype=np.uint8)

        # 遍历所有颜色类别，标注建筑受损情况
        for damage_color, overlay_color in damage_overlay_colors.items():
            mask_bin = (mask_img[:, :, 0] == damage_color[0]) & \
                       (mask_img[:, :, 1] == damage_color[1]) & \
                       (mask_img[:, :, 2] == damage_color[2])
            overlay[mask_bin] = overlay_color  # 赋值颜色（RGBA）

        # 将叠加图层与原图合并
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        image_result = np.uint8(image_post * (1 - alpha) + overlay[:, :, :3] * alpha)

        # 保存结果图像
        temp_path = save_image(image_result)

        return temp_path

    def _count_building_damage(self, mask_img: np.ndarray) -> dict:
        """
        统计不同损伤程度的建筑物数量

        :param mask_img: 变化检测结果，掩码图像
        :return: 受损统计结果 (字典)
        """
        # 统计各类别的建筑物数量
        damage_counts = OrderedDict({key: 0 for key in  self.color_map.keys()})

        # 遍历每种损伤类别
        for damage_type, color in  self.color_map.items():
            # 生成二值掩码
            mask_bin = cv2.inRange(mask_img, np.array(color), np.array(color))

            # 进行连通域分析，获取独立建筑数量
            num, _, _, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

            # 记录建筑数量（去掉背景区域）
            damage_counts[damage_type] = num - 1 if num > 1 else 0

        return damage_counts

    def execute(self, post_img_path: str, change_detection_mask_path: str):
        """
        对灾后建筑物受损情况进行分析和统计

        :param post_img_path: 受灾后的遥感图像路径
        :param change_detection_mask_path: 受灾前后变化检测的掩码图路径
        """
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        post_img_path = os.path.join(root_path, post_img_path)
        change_detection_mask_path = os.path.join(root_path, change_detection_mask_path)

        img_post = load_image(post_img_path)
        mask_img = load_image(change_detection_mask_path)

        damage_counts = self._count_building_damage(mask_img)
        statistics_img_path = self._visualize_damage_statistics(damage_counts)
        damage_img_path = self._visualize_damage(img_post, mask_img)

        # 构造返回值
        message = {"role": "tool",
                   "name": self.name,
                   "content": [
                        {'type': 'text', 'text': f'{dict(damage_counts)}', 'text_description': '灾后建筑物受损情况统计'},
                        {'type': 'image', 'image_data': f'{statistics_img_path}', 'image_description': '灾后建筑物受损情况统计可视化柱状图'},
                        {'type': 'image', 'image_data': f'{damage_img_path}', 'image_description': '灾后建筑物受损情况可视化影像图'}
                   ]}

        return message


if __name__ == '__main__':
    # 示例
    mask_path = r'../../demo_data/change_detection_mask.png'
    img_path = r'../../demo_data/post.png'

    quantity_calculation = QuantityCalculation()
    result = quantity_calculation.execute(mask_path, img_path)

    print(result)