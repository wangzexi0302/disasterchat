import base64
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from io import BytesIO
from PIL import Image
import os

logger = logging.getLogger(__name__)

class ChangeDetectionModel:
    """
    遥感图像变化检测模型
    处理两张时序卫星图像(前后)，检测变化区域
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        初始化变化检测模型
        
        Args:
            model_path: 模型权重文件路径，默认为None，使用内置模型
            device: 计算设备，'cpu'或'cuda'
        """
        self.model_path = model_path
        self.device = device
        self.change_types = ["新增建筑", "植被减少", "水体变化", "道路变化", "其他变化"]
        logger.info(f"变化检测模型初始化，设备: {device}")
        
        # 实际项目中，这里应该加载真实的模型
        # self.model = load_actual_model(model_path, device)
        logger.info("变化检测模型加载完成")
    
    def _preprocess_images(self, pre_image_data: str, post_image_data: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理前后两张图像数据
        
        Args:
            pre_image_data: 变化前图像的base64编码
            post_image_data: 变化后图像的base64编码
            
        Returns:
            处理后的两张图像的numpy数组
        """
        # 处理前图像
        if "base64," in pre_image_data:
            pre_image_data = pre_image_data.split("base64,")[1]
        pre_image_bytes = base64.b64decode(pre_image_data)
        pre_image = Image.open(BytesIO(pre_image_bytes))
        pre_array = np.array(pre_image)
        
        # 处理后图像
        if "base64," in post_image_data:
            post_image_data = post_image_data.split("base64,")[1]
        post_image_bytes = base64.b64decode(post_image_data)
        post_image = Image.open(BytesIO(post_image_bytes))
        post_array = np.array(post_image)
        
        # 检查尺寸一致性
        if pre_array.shape[:2] != post_array.shape[:2]:
            logger.warning("前后图像尺寸不一致，进行调整")
            # 调整图像尺寸到相同大小
            height, width = min(pre_array.shape[0], post_array.shape[0]), min(pre_array.shape[1], post_array.shape[1])
            pre_array = pre_array[:height, :width]
            post_array = post_array[:height, :width]
        
        return pre_array, post_array
    
    def _postprocess_result(self, change_mask: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        后处理模型输出
        
        Args:
            change_mask: 模型预测的变化掩码
            
        Returns:
            变化检测结果图像的base64编码和变化统计信息
        """
        # 模拟生成彩色变化检测图
        height, width = change_mask.shape[:2]
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 根据不同变化类型设置不同颜色 (实际应用中替换为正确的可视化逻辑)
        # 0: 无变化, 1: 新增建筑, 2: 植被减少, 3: 水体变化, 4: 道路变化, 5: 其他变化
        color_map = {
            0: [0, 0, 0],       # 黑色: 无变化
            1: [255, 0, 0],     # 红色: 新增建筑
            2: [0, 255, 0],     # 绿色: 植被减少
            3: [0, 0, 255],     # 蓝色: 水体变化
            4: [255, 255, 0],   # 黄色: 道路变化
            5: [255, 0, 255]    # 紫色: 其他变化
        }
        
        # 模拟结果统计
        total_pixels = height * width
        changed_pixels = np.sum(change_mask > 0)
        change_percentage = changed_pixels / total_pixels
        
        # 模拟各类变化统计
        changes_by_type = {
            "新增建筑": np.sum(change_mask == 1) / total_pixels * 100,
            "植被减少": np.sum(change_mask == 2) / total_pixels * 100,
            "水体变化": np.sum(change_mask == 3) / total_pixels * 100,
            "道路变化": np.sum(change_mask == 4) / total_pixels * 100,
            "其他变化": np.sum(change_mask == 5) / total_pixels * 100
        }
        
        # 将numpy数组转换为Image
        result_image = Image.fromarray(colored_mask)
        
        # 保存为bytes
        buffer = BytesIO()
        result_image.save(buffer, format="PNG")
        
        # 转换为base64
        change_map_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 构造变化统计信息
        change_statistics = {
            "total_changed_area": changed_pixels / 1000000,  # 模拟平方公里
            "change_percentage": change_percentage,
            "changes_by_type": changes_by_type
        }
        
        return change_map_b64, change_statistics
    
    def detect_change(self, pre_image_data: str, post_image_data: str, 
                     sensitivity: float = 0.5,
                     change_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行变化检测
        
        Args:
            pre_image_data: 变化前图像的base64编码数据
            post_image_data: 变化后图像的base64编码数据
            sensitivity: 变化检测敏感度，0-1之间
            change_types: 需要检测的变化类型列表，默认为全部类型
            
        Returns:
            变化检测结果，包括变化图和统计信息
        """
        logger.info("开始执行变化检测")
        start_time = time.time()
        
        try:
            # 预处理图像
            pre_array, post_array = self._preprocess_images(pre_image_data, post_image_data)
            
            # 实际项目中，这里应该将图像传递给真实的模型进行推理
            # change_mask = self.model(pre_array, post_array, sensitivity)
            
            # 模拟预测结果 - 在实际实现中替换为真实模型输出
            height, width = pre_array.shape[:2]
            change_mask = np.random.randint(0, 6, (height, width))  # 0-5的随机值，模拟不同类型的变化
            
            # 根据sensitivity调整变化区域比例
            threshold = 1.0 - sensitivity
            change_mask[np.random.random((height, width)) < threshold] = 0
            
            # 后处理结果
            change_map, change_statistics = self._postprocess_result(change_mask)
            
            processing_time = time.time() - start_time
            
            result = {
                "change_map": change_map,
                "change_statistics": change_statistics,
                "processing_time": processing_time,
                "confidence_score": 0.85 + 0.1 * sensitivity  # 模拟置信度
            }
            
            logger.info(f"变化检测完成，处理时间: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"变化检测处理失败: {str(e)}", exc_info=True)
            raise
