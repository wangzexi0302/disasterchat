"""
响应模板配置模块
包含预定义的对话响应模板，用于特定场景的响应生成
"""

from typing import Dict, List, Any, Optional

class ResponseTemplate:
    """响应模板类，定义了生成响应所需的所有参数"""
    def __init__(
        self, 
        delay: int = 5, 
        text: str = "", 
        images: List[str] = None,
        image_type: str = "post"
    ):
        self.delay = delay
        self.text = text
        self.images = images or []
        self.image_type = image_type

# 模板响应配置字典
TEMPLATE_RESPONSES: Dict[str, ResponseTemplate] = {
    # 样例图片一
    "请告诉我灾后影像的大致受灾情况": ResponseTemplate(
        delay=7,
        text="""这张影像显示了一个受灾区域，飓风后的卫星图像。从图像来看，主要的受灾情况包括:
大面积积水:图像显示大片的浑浊水域，覆盖了树林和部分居民区。这表明该区域可能经历了严重的洪水，导致陆地被淹没。
居民区受灾:部分房屋仍然可见，但许多看起来被水包围或部分淹没，这可能导致基础设施受损、居民被困或者财产损失。
树木和植被受影响:尽管树木仍然茂密，但被水淹没的情况可能导致植被根部受损，长期来看可能影响生态系统。
道路情况不明:由于洪水的覆盖，难以判断道路是否完好或者是否仍可通行，可能影响救援和疏散行动。"""
    ),
    "请判断受灾后A点到B点的道路是否通畅": ResponseTemplate(
        delay=13,
        text="根据您所提供的图像，经过路径判断受灾后A点B点之间的道路受到灾害影响不通畅。",
        images=["test_image_3.png"],
        image_type="post"
    ),
    "那受灾前A点到B点的道路是否通畅呢？": ResponseTemplate(
        delay=12,
        text="根据您所提供的图像，经过路径判断受灾前A点B点之间的道路是通畅的。",
        images=["test_image_4.png"],
        image_type="post"
    ),
    "请根据受灾场景综合判断房屋受损情况，要求尽可能的详细，且提供受灾图像的基本信息。": ResponseTemplate(
        delay=20,
        text="""这张卫星图像展示了飓风过后一个受灾区域的全貌。图像中可以明显看到大片浑浊的水域覆盖了树林和部分居民区，表明该区域经历了严重洪水，陆地大面积被淹。部分房屋依然可辨，但许多建筑似乎被水包围或部分浸泡，暗示基础设施可能遭受破坏，居民也可能面临被困和财产损失的风险。虽然树木依旧茂密，但被淹的情况可能对植被的根系造成损伤，长期来看会影响生态系统；而由于洪水覆盖，难以判断道路的完好性和通行状况，这可能对救援和疏散行动构成阻碍。根据对36个建筑物的统计，数据显示有22个建筑物无损坏，主要分布在图像下半部分和右侧；4个建筑物显示轻微损坏，分散在图像右下部；7个建筑物受严重损坏，主要集中在图像的中部；另外还有4个建筑物无法识别。这种分布表明，虽然大部分房屋没有明显损坏，但局部区域尤其是图像中下部和左侧，受灾情况较为严重，提示救援和恢复工作需针对性展开。""",
        images=["test_image_5.png"],
        image_type="post"
    ),
    "请详细介绍灾害后各地物的面积统计": ResponseTemplate(
        delay=7,
        text="""根据灾后影像的分割结果，各个地物的面积如下： \n1. 建筑物：66260.0 平方米。 \n2. 水体：546520.0 平方米。 \n3. 道路：118720.0平方米""",
        images=["test_image_3.png"],
        image_type="post"
    ),
    "请在灾害前影像中规划A、B两点的具体通行路径": ResponseTemplate(
        delay=10,
        text="根据您所提供的图像，A、B两点间的具体路径如下：",
        images=["path_pic.jpg", "path_mask.jpg"],
        image_type="post"
    ),
    # 样例图片二
    
}

# 预设提示模板
PREDEFINED_TEMPLATES = [
    {
        "id": "disaster_summary",
        "name": "分析整体灾情",
        "content": "请告诉我灾后影像的大致受灾情况"
    },
    {
        "id": "relief_suggestion",
        "name": "救援建议",
        "content": "请提供具体的灾后救援建议和资源分配方案"
    }
]
