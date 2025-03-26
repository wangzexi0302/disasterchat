# -*- coding: utf-8 -*-
import tempfile
from collections import OrderedDict
from app.tools.base import Tool
from typing import Dict, Any
import requests
import cv2
import numpy as np
from utils import load_image, save_image, print_mask_color
import matplotlib.pyplot as plt


class AreaCalculation(Tool):
    """面积计算工具"""

    def __init__(self):
        # 各类型地物及颜色
        self.building_color_map = OrderedDict({
            # "no-damage": (0, 255, 0),  # 无损伤
            "minor-damage": (0, 255, 255),  # 轻度损伤
            "major-damage": (0, 0, 255),  # 重大损伤
            # "un-classified": (255, 0, 255),  # 未分类
        })
        self.water_color_map = OrderedDict({
            # "buildings": (0, 0, 255),  # 建筑物
            # "road": (128, 128, 128),  # 道路
            "water": (255, 0, 0),  # 水体
        })
        self.road_color_map = OrderedDict({
            "road": (128, 128, 128),  # 道路
        })

    @property
    def name(self) -> str:
        return "are_calculation"

    @property
    def description(self) -> str:
        return "对水体、建筑物、道路等的受灾面积进行统计"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "post_img_path": {
                "type": "string",
                "description": "灾后遥感影像图",
                "required": True
            },
            "change_detection_mask_path": {
                "type": "string",
                "description": "变化检测掩码图",
                "required": True
            },
            "pre_segmentation_mask_path": {
                "type": "string",
                "description": "灾前语义分割掩码图",
                "required": True
            },
            "post_segmentation_mask_path": {
                "type": "string",
                "description": "灾后语义分割掩码图",
                "required": True
            }
        }

    def _visualize_damage(self, image_post: np.ndarray, building_mask_img: np.ndarray,
                          water_mask_img: np.ndarray, road_mask_img: np.ndarray) -> str:
        """
        对受灾区域进行可视化

        :param image_post: 灾后影像
        :param building_mask_img: 灾后语义分割建筑物掩码影像
        :param water_mask_img: 灾后语义分割水体掩码影像
        :param road_mask_img: 灾后语义分割道路掩码影像
        :return: 可视化结果路径
        """
        # 创建一个与原图大小相同的透明图层
        overlay = np.zeros((image_post.shape[0], image_post.shape[1], 4), dtype=np.uint8)

        # 标注建筑受损情况
        for _, damage_color in self.building_color_map.items():
            mask_bin = (building_mask_img[:, :, 0] == damage_color[0]) & \
                       (building_mask_img[:, :, 1] == damage_color[1]) & \
                       (building_mask_img[:, :, 2] == damage_color[2])
            overlay[mask_bin] = (0, 0, 255, 192)  # 赋值颜色（RGBA）

        # 标注道路受灾情况
        for _, damage_color in self.road_color_map.items():
            mask_bin = (road_mask_img[:, :, 0] == damage_color[0]) & \
                       (road_mask_img[:, :, 1] == damage_color[1]) & \
                       (road_mask_img[:, :, 2] == damage_color[2])
            overlay[mask_bin] = (128, 128, 128, 192)  # 赋值颜色（RGBA）

        # 标注水体情况
        for _, damage_color in self.water_color_map.items():
            mask_bin = (water_mask_img[:, :, 0] == damage_color[0]) & \
                       (water_mask_img[:, :, 1] == damage_color[1]) & \
                       (water_mask_img[:, :, 2] == damage_color[2])
            overlay[mask_bin] = (255, 0, 0, 192)  # 赋值颜色（RGBA）

        # 将叠加图层与原图合并
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        image_result = np.uint8(image_post * (1 - alpha) + overlay[:, :, :3] * alpha)

        # 保存结果图像
        temp_path = save_image(image_result)

        return temp_path

    def _damage_area(self, mask_img: np.ndarray, gsd: float, color_map) -> dict:
        """
        统计各类地物的损伤面积

        :param mask_img: 掩码图像
        :param gsd: 地面分辨率
        :return: 受损面积统计结果 (字典)
        """
        # 统计各类别的建筑物数量
        damage_areas = OrderedDict({key: 0 for key in color_map.keys()})

        # 遍历每种损伤类别
        for damage_type, color in color_map.items():
            # 生成二值掩码
            mask_bin = cv2.inRange(mask_img, np.array(color), np.array(color))

            # 进行连通域分析，获取独立建筑数量
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

            # 计算地物数量（去掉背景区域）
            num = num - 1 if num > 1 else 0

            # 计算总受灾面积（平方米）
            total_area_pixels = np.sum(stats[1:, cv2.CC_STAT_AREA])  # 排除背景区域
            total_area_meters = float(total_area_pixels * (gsd ** 2))  # 像素面积转换为平方米

            # 记录面积
            damage_areas[damage_type] = (num, total_area_meters)

        return damage_areas

    def execute(self,pos_img_path: str, change_detection_mask_path: str,
                pre_segmentation_mask_path: str, post_segmentation_mask_path: str):
        """
        计算各类地物的受灾面积，并进行可视化。

        :param change_detection_mask_path: 建筑物变化检测的掩码图像路径
        :param pre_segmentation_mask_path: 灾前语义分割的掩码图像路径
        :param post_segmentation_mask_path: 灾后语义分割的掩码图像路径
        :param pos_img_path: 受灾后的遥感图像路径
        """
        img_post = load_image(pos_img_path)
        change_detection_mask_img = load_image(change_detection_mask_path)
        pre_segmentation_mask_img = load_image(pre_segmentation_mask_path)
        post_segmentation_mask_img = load_image(post_segmentation_mask_path)

        # 计算各类地物灾前、灾后的面积
        # 1. 建筑物
        building_damage_area = self._damage_area(change_detection_mask_img, gsd=2, color_map=self.building_color_map)
        building_damage_area = building_damage_area['minor-damage'][1] + building_damage_area['major-damage'][1]

        # 2. 水体
        water_damage_area = self._damage_area(post_segmentation_mask_img, gsd=2, color_map=self.water_color_map)
        water_damage_area = water_damage_area['water'][1]

        # 3. 道路
        road_color = self.road_color_map['road']
        # 提取道路二值掩膜 (灾前/灾后)
        pre_road = np.all(pre_segmentation_mask_img == road_color, axis=-1)
        post_road = np.all(post_segmentation_mask_img == road_color, axis=-1)
        # 计算受损道路区域 (灾前有道路但灾后无)
        road_damage_mask = pre_road & ~post_road
        overlay = np.zeros_like(post_segmentation_mask_img, dtype=np.uint8)
        overlay[road_damage_mask] = road_color
        road_damage_area = self._damage_area(overlay, gsd=2, color_map=self.road_color_map)
        road_damage_area= road_damage_area['road'][1]

        damage_area = {'building': building_damage_area, 'water': water_damage_area, 'road': road_damage_area}
        print(damage_area)

        # 灾后受损可视化
        damage_img_path = self._visualize_damage(img_post, change_detection_mask_img, post_segmentation_mask_img, overlay)

        return damage_area, damage_img_path


if __name__ == '__main__':
    # 示例
    change_detection_mask_path = r'../../demo_data/demo_1/change_detection_mask.png'
    pre_segmentation_mask_path = r'../../demo_data/demo_1/pre_segmentation_mask.png'
    post_segmentation_mask_path = r'../../demo_data/demo_1/post_segmentation_mask.png'
    post_img_path = r'../../demo_data/demo_1/post.png'

    area_calculation = AreaCalculation()
    result = area_calculation.execute(post_img_path, change_detection_mask_path,
                                      pre_segmentation_mask_path, post_segmentation_mask_path)

    print(result)