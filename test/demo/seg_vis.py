import os
import cv2
import numpy as np
import json
import base64
import io
from PIL import Image, ImageDraw

def draw_polygon(draw, points, color):
    """绘制多边形"""
    # 将points转换为元组列表
    points = [(float(x), float(y)) for x, y in points]
    draw.polygon(points, outline=color, fill=color)

def process_mask(mask_data, points, color, mask_img):
    """处理mask类型的标注"""
    # 解码Base64数据
    mask_bytes = base64.b64decode(mask_data)
    # 将字节数据转换为图像
    mask = Image.open(io.BytesIO(mask_bytes))
    mask = mask.convert('L')  # 转换为灰度图像
    
    # 获取mask的位置信息
    x1, y1 = points[0]
    x2, y2 = points[1]
    width = int(x2 - x1)
    height = int(y2 - y1)
    
    # 调整mask大小以匹配points定义的区域
    mask = mask.resize((width, height))
    
    # 创建临时绘图对象
    temp_draw = ImageDraw.Draw(mask_img)
    
    # 将mask放置在正确的位置
    for y in range(height):
        for x in range(width):
            if mask.getpixel((x, y)) > 0:  # 如果掩码点有值
                temp_draw.point((x1 + x, y1 + y), fill=color)

def main():
    # 文件路径设置
    json_path = '3/post.json'
    image_path = '3/post.png'
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # 检查文件是否存在
    if not os.path.exists(json_path) or not os.path.exists(image_path):
        print(f"文件不存在: {json_path} 或 {image_path}")
        return

    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 读取原始图像
    image = Image.open(image_path)
    width, height = image.size

    # 创建掩码图像
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # 颜色映射
    label_colors = {
        'water': 1,
        'road': 2,
        'building': 3
    }

    # 绘制所有多边形和掩码
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        shape_type = shape.get('shape_type', 'polygon')  # 默认为polygon类型
        
        if label in label_colors:
            color = label_colors[label]
            
            if shape_type == 'polygon':
                draw_polygon(draw, points, color)
            elif shape_type == 'mask':
                mask_data = shape.get('mask', '')
                if mask_data:
                    process_mask(mask_data, points, color, mask)

    # 将掩码转换为numpy数组
    mask_array = np.array(mask)

    # 可视化结果
    # 为每个类别设置不同的颜色
    color_map = np.array([
        [0, 0, 0],        # 背景 (黑色)
        [0, 0, 255],      # 水域 (蓝色)
        [128, 128, 128],  # 道路 (灰色)
        [255, 0, 0]       # 建筑 (红色)
    ], dtype=np.uint8)

    # 创建彩色掩码
    color_mask = color_map[mask_array]

    # 将原始图像转换为numpy数组
    image_array = np.array(image)

    # 混合原始图像和掩码
    alpha = 0.5
    result = cv2.addWeighted(image_array, 1-alpha, color_mask, alpha, 0)

    # 保存结果
    output_path = os.path.join(output_dir, '1_pol_visualized.png')
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # 同时保存语义分割掩码
    mask_output_path = os.path.join(output_dir, '1_pol_mask.png')
    cv2.imwrite(mask_output_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    print(f"可视化结果已保存至: {output_path}")
    print(f"语义分割掩码已保存至: {mask_output_path}")

if __name__ == "__main__":
    main()