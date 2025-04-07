# -*- coding: utf-8 -*-
from app.tools.base import Tool
from typing import Dict, Any
import os
import cv2
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
<<<<<<< HEAD
from .utils import load_image, save_image
=======
from app.tools.utils import load_image, save_image
>>>>>>> c90fc095383ff1b2ab32ad4099076c36dc63cf08


class PathCalculation(Tool):
    """路径计算工具"""

    @property
    def name(self) -> str:
        return "path_calculation"

    @property
    def description(self) -> str:
        return "评估给定起始点的道路是否可通行"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "pre_or_post_img_path": {
                "type": "string",
                "description": "灾前或灾后遥感影像图路径",
                "required": True
            },
            "pre_or_post_segmentation_mask_path": {
                "type": "string",
                "description": "灾前或灾后语义分割掩码图路径",
                "required": True
            },
            "point_A": {
                "type": "tuple",
                "description": "用户提供的起始点坐标",
                "required": True
            },
            "point_B": {
                "type": "tuple",
                "description": "用户提供的终止点坐标",
                "required": True
            }
        }

    def _load_mask_image(self, mask_path, img_path):
        """
        读取语义分割 mask 图片，并提取道路区域。

        :param mask_path: 掩码图片路径
        :return: 道路二值化掩码 (255 = 道路, 0 = 其他)
        """
        mask = load_image(mask_path)  # 读取 3 通道颜色图
        road_mask = np.zeros(mask.shape[:2], dtype=np.uint8)  # 创建空白掩码

        # 识别道路区域（灰色: [128, 128, 128]）
        road_pixels = (mask[:, :, 0] == 128) & (mask[:, :, 1] == 128) & (mask[:, :, 2] == 128)
        road_mask[road_pixels] = 255  # 设置道路为白色（可通行区域）

        img = load_image(img_path)  # 读取 3 通道颜色图

        return road_mask, mask, img  # 返回掩码，mask 图，原始图

    def _build_graph_from_mask(self, road_mask):
        """
        根据道路二值掩码构建网络图

        :param road_mask: 二值化道路掩码
        :return: networkx 无向图
        """
        height, width = road_mask.shape
        G = nx.Graph()

        for y in range(height):
            for x in range(width):
                if road_mask[y, x] == 255:  # 只添加道路上的点
                    G.add_node((x, y))

                    # 连接 8 邻域的可通行像素
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < width and 0 <= ny_ < height and road_mask[ny_, nx_] == 255:
                            G.add_edge((x, y), (nx_, ny_))  # 连接两个相邻的道路点

        return G

    def _find_nearest_road_point(self, road_mask, point):
        """
        在道路掩码上找到离给定点最近的道路像素

        :param road_mask: 二值化道路掩码
        :param point: 用户输入点 (x, y)
        :return: 最近的道路像素 (x, y)
        """
        y_indices, x_indices = np.where(road_mask == 255)  # 获取所有道路像素
        road_pixels = np.column_stack((x_indices, y_indices))  # 组成 (x, y) 形式

        if len(road_pixels) == 0:
            return None  # 没有道路

        tree = KDTree(road_pixels)
        distance, index = tree.query(point)  # 找到最近的道路像素

        return tuple(road_pixels[index])  # 返回最近的道路像素点 (x, y)

    def _visualize_path(self, image, path, point_A, point_B):
        """
        在原始图像上绘制通行路径。

        :param image: 原始 mask 图像
        :param path: 计算出的通行路径
        :param point_A: 起点 (x, y)
        :param point_B: 终点 (x, y)
        """
        if path:
            # 绘制路径（黄色）
            for i in range(len(path) - 1):
                cv2.line(image, path[i], path[i + 1], (255, 255, 0), 2)

        # 绘制起点 A（绿色）
        cv2.circle(image, point_A, 5, (0, 255, 0), -1)

        # 绘制终点 B（红色）
        cv2.circle(image, point_B, 5, (0, 0, 255), -1)

        # 保存结果
        output_path = save_image(image)
        # print(f"路径可视化已保存至: {output_path}")

        return output_path

    def _check_path_accessibility(self, mask_path, img_path, point_A, point_B, gsd=2):
        """
        评估 A 点到 B 点是否可通行，并可视化路径。

        :param mask_path: 语义分割的掩码图路径
        :param img_path: 原始图像路径
        :param point_A: 用户输入的 A 点 (像素坐标)
        :param point_B: 用户输入的 B 点 (像素坐标)
        :return:
            accessibility: 可通行 or 不可通行
            mask_output_path: 语义分割的掩码图可视化路径保存位置
            img_out_path: 原始图像可视化路径保存位置
        """
        # 读取 mask 图像并提取道路区域
        road_mask, mask_img, ori_img = self._load_mask_image(mask_path, img_path)

        # 构建道路连通性图
        road_graph = self._build_graph_from_mask(road_mask)

        # 将 A、B 点映射到最近的道路像素
        point_A_nearest = self._find_nearest_road_point(road_mask, point_A)
        point_B_nearest = self._find_nearest_road_point(road_mask, point_B)

        # 检查路径通行性
        if point_A_nearest and point_B_nearest:
            if point_A_nearest in road_graph and point_B_nearest in road_graph:
                try:
                    # A* 搜索算法
                    path = nx.astar_path(road_graph, source=point_A_nearest, target=point_B_nearest)

                    # 可视化路径
                    mask_output_path = self._visualize_path(mask_img, path, point_A_nearest, point_B_nearest)
                    img_out_path = self._visualize_path(ori_img, path, point_A_nearest, point_B_nearest)

                    return f"A 到 B 可通行，路径长度: {len(path)*gsd} 米", mask_output_path, img_out_path
                except nx.NetworkXNoPath:
                    # 可视化A、B点
                    mask_output_path = self._visualize_path(mask_img, None, point_A_nearest, point_B_nearest)
                    img_out_path = self._visualize_path(ori_img, None, point_A_nearest, point_B_nearest)

                    return "A 到 B 不可通行", mask_output_path, img_out_path
            else:
                # 可视化A、B点
                mask_output_path = self._visualize_path(mask_img, None, point_A_nearest, point_B_nearest)
                img_out_path = self._visualize_path(ori_img, None, point_A_nearest, point_B_nearest)

                return "A 或 B 不在道路范围内，无法评估", mask_output_path, img_out_path
        else:
            return "A 或 B 无法映射到道路像素，无法评估", None, None

    def execute(self, pre_or_post_img_path: str, pre_or_post_segmentation_mask_path: str, point_A: tuple, point_B: tuple):
        """
        评估 A 点到 B 点是否可通行，并可视化路径。

        :param pre_or_post_img_path: 遥感影像路径
        :param pre_or_post_segmentation_mask_path: 语义分割的掩码图路径
        :param point_A: 用户输入的 A 点 (像素坐标)
        :param point_B: 用户输入的 B 点 (像素坐标)
        """
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        img_path = os.path.join(root_path, pre_or_post_img_path)
        mask_path = os.path.join(root_path, pre_or_post_segmentation_mask_path)

        acc_res, mask_vis_path, img_vis_path = self._check_path_accessibility(mask_path, img_path, point_A, point_B)

        # 构造返回值
        message = {"role": "tool",
                   "name": self.name,
                   "content": [
                        {'type': 'text', 'text': f'{acc_res}', 'text_description': '道路通行情况'},
                        {'type': 'image', 'image_data': f'{mask_vis_path}', 'image_description': '道路通行情况可视化掩码图'},
                        {'type': 'image', 'image_data': f'{img_vis_path}', 'image_description': '道路通行情况可视化影像图'}]}

        return message

if __name__ == '__main__':
    # 示例
    mask_path = 'demo_data/pre_segmentation_mask.png'
    img_path = 'demo_data/pre.png'
    point_A = (400, 600)  # 用户指定的 A 点（像素坐标）
    point_B = (900, 1000)  # 用户指定的 B 点（像素坐标）

    path_calculation = PathCalculation()
    result = path_calculation.execute(img_path, mask_path, point_A, point_B)

    print(result)