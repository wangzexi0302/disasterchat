from app.tools.path_calculation import PathCalculation
from app.tools.area_calculation import AreaCalculation
from app.tools.quantity_calculation import QuantityCalculation
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_tools():
    # Test case 1: Test PathCalculation
    logger.info("测试【路径计算工具：评估A到B是否可达】...")
    mask_path = 'demo_data/pre_segmentation_mask.png'
    img_path = 'demo_data/pre.png'
    point_A = (400, 600)  # 用户指定的 A 点（像素坐标）
    point_B = (900, 1000)  # 用户指定的 B 点（像素坐标）

    path_calculation = PathCalculation()
    result = path_calculation.execute(img_path, mask_path, point_A, point_B)
    logger.info(result)

    # Test case 2: Test AreaCalculation
    logger.info("测试【面积计算工具：各类地物受灾面积统计】...")
    change_detection_mask_path = 'demo_data/change_detection_mask.png'
    pre_segmentation_mask_path = 'demo_data/pre_segmentation_mask.png'
    post_segmentation_mask_path = 'demo_data/post_segmentation_mask.png'
    post_img_path = r'demo_data/post.png'

    area_calculation = AreaCalculation()
    result = area_calculation.execute(post_img_path, change_detection_mask_path,
                                      pre_segmentation_mask_path, post_segmentation_mask_path)
    logger.info(result)

    # Test case 3: Test QuantityCalculation
    logger.info("测试【数量计算工具：建筑物损伤统计】...")
    mask_path = 'demo_data/change_detection_mask.png'
    img_path = 'demo_data/post.png'

    quantity_calculation = QuantityCalculation()
    result = quantity_calculation.execute(img_path, mask_path)
    logger.info(result)

test_tools()
