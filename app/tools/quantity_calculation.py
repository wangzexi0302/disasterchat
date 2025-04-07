# -*- coding: utf-8 -*-
import tempfile
from collections import OrderedDict
from app.tools.base import Tool
from typing import Dict, Any
import requests
import cv2
import numpy as np
from .utils import load_image, save_image, print_mask_color
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
        return "对受灾建筑物各类受损情况进行统计"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "img_path": {
                "type": "string",
                "description": "遥感影像图",
                "required": True
            },
            "mask_path": {
                "type": "string",
                "description": "语义分割掩码图",
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
        print(f"路径可视化已保存至: {temp_path}")
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

    def execute(self, img_path: str, mask_path: str):
        """
        评估 A 点到 B 点是否可通行，并可视化路径。

        :param img_path: 受灾后的遥感图像路径
        :param mask_path: 语义分割的掩码图路径
        """
        img_post = load_image(img_path)
        mask_img = load_image(mask_path)

        damage_counts = self._count_building_damage(mask_img)
        statistics_img_path = self._visualize_damage_statistics(damage_counts)
        damage_img_path = self._visualize_damage(img_post, mask_img)

        return damage_counts, statistics_img_path, damage_img_path


if __name__ == '__main__':
    # 示例
    mask_path = r'../../demo_data/demo_1/change_detection_mask.png'
    img_path = r'../../demo_data/demo_1/post.png'

    quantity_calculation = QuantityCalculation()
    result = quantity_calculation.execute(mask_path, img_path)

    print(result)