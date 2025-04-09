import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import re

def wkt_to_coords(wkt_string):
    """从WKT字符串提取坐标"""
    # 使用正则表达式提取坐标对
    pattern = r'(\d+\.?\d*)\s+(\d+\.?\d*)'
    coords = re.findall(pattern, wkt_string)
    # 转换为浮点数对
    coords = [(float(x), float(y)) for x, y in coords]
    return coords

def draw_polygon(image, coords, color, thickness=2, filled=False):
    """在图像上绘制多边形"""
    if filled:
        cv2.fillPoly(image, [np.array(coords, dtype=np.int32)], color)
    else:
        cv2.polylines(image, [np.array(coords, dtype=np.int32)], True, color, thickness)
    return image

def main():
    # 设置文件路径
    json_path = '3/hurricane-harvey_00000312_post_disaster.json'
    image_path = '3/hurricane-harvey_00000312_post_disaster.png'
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"JSON文件不存在: {json_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        return

    # 读取JSON数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 创建可视化图像
    vis_image = image.copy()
    mask_image = np.zeros_like(image)
    
    # 定义不同损坏类别的颜色
    damage_colors = {
        "no-damage": (0, 255, 0),      # 绿色
        "minor-damage": (0, 255, 255),  # 黄色
        "major-damage": (0, 0, 255),    # 红色
        "un-classified": (255, 0, 255)  # 紫色
    }
    
    # 提取特征并绘制
    features = data.get("features", {}).get("xy", [])
    building_counts = {
        "no-damage": 0,
        "minor-damage": 0,
        "major-damage": 0,
        "un-classified": 0
    }
    
    for feature in features:
        properties = feature.get("properties", {})
        feature_type = properties.get("feature_type")
        subtype = properties.get("subtype")
        
        # 只处理建筑物
        if feature_type == "building":
            wkt = feature.get("wkt", "")
            coords = wkt_to_coords(wkt)
            
            if subtype in damage_colors:
                color = damage_colors[subtype]
                # 在可视化图像上绘制轮廓
                vis_image = draw_polygon(vis_image, coords, color, thickness=2)
                # 在掩码图像上填充区域
                mask_image = draw_polygon(mask_image, coords, color, filled=True)
                building_counts[subtype] += 1
    
    # 创建统计图
    labels = list(building_counts.keys())
    values = list(building_counts.values())
    colors = [damage_colors[label] for label in labels]
    rgb_colors = [(r/255, g/255, b/255) for (b, g, r) in colors]  # 转换BGR到RGB

    plt.figure(figsize=(12, 5))
    
    # 左侧显示图像
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title('建筑物损坏分类')
    plt.axis('off')
    
    # 右侧显示统计图
    plt.subplot(1, 2, 2)
    bars = plt.bar(labels, values, color=rgb_colors)
    plt.title('建筑物损坏统计')
    plt.xlabel('损坏类别')
    plt.ylabel('数量')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom')
    
    # 保存可视化结果
    output_path = os.path.join(output_dir, 'damage_visualization.png')
    plt.tight_layout()
    plt.savefig(output_path)
    
    # 保存带轮廓的图像
    cv2.imwrite(os.path.join(output_dir, 'building_outlines.png'), vis_image)
    
    # 保存掩码图像
    cv2.imwrite(os.path.join(output_dir, 'damage_mask.png'), mask_image)
    
    # 创建半透明叠加图
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1-alpha, mask_image, alpha, 0)
    cv2.imwrite(os.path.join(output_dir, 'overlay_visualization.png'), overlay)
    
    print(f"可视化结果已保存至: {output_dir}")
    print(f"建筑物统计: {building_counts}")

if __name__ == "__main__":
    main()