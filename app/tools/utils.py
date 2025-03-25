import cv2
import numpy as np
import base64
import tempfile


def print_mask_color(mask_path):
    mask = load_image(mask_path)
    unique_colors = np.unique(mask.reshape(-1, 3), axis=0)

    for color in unique_colors:
        print(f"Color: {color}")

def load_image(image_path: str) -> np.ndarray:
    """
    加载影像
    :param image_path: 影像路径
    :return: 影像数据
    """

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法加载影像: {image_path}")
    return image


def process_image(image_data: str) -> str:
    """
    处理base64编码的图像数据

    Args:
        image_data: base64编码的图像数据

    Returns:
        临时图像文件的路径
    """
    # 去除可能存在的base64前缀
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]

    temp_files = []
    try:
        # 解码base64图像数据
        image_bytes = base64.b64decode(image_data)
        # 创建临时文件保存图像数据
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
            temp_files.append(temp_path)  # 记录临时文件路径，以便后续删除
        return temp_path
    except Exception as e:
        raise ValueError("处理图像数据失败！")


def load_image_as_base64(image_path):
    """将图像文件加载为base64编码字符串"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        return None


def save_image(image_data: np.ndarray):
    """将图像数据保存为临时文件"""
    temp_files = []
    try:
        # 创建临时文件保存结果图像
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image_data)
            temp_files.append(temp_path)
        return temp_files
    except Exception as e:
        raise ValueError("保存图像数据失败！")
