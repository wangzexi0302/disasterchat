# 此文件用于标记工具目录为Python包
from app.tools.base import Tool
from app.tools.weather import GetWeatherTool
from app.tools.disaster_info import GetDisasterInfoTool
from app.tools.call_multimodel import CallMultiModel
from app.tools.area_calculation import AreaCalculation
from app.tools.path_calculation import PathCalculation
from app.tools.quantity_calculation import QuantityCalculation


# 导出所有可用工具
available_tools = [
    GetWeatherTool(),
    GetDisasterInfoTool(),
    AreaCalculation(),
    PathCalculation(),
    QuantityCalculation()
    ]

models = [
    CallMultiModel()
]

__all__ = ["Tool", "available_tools", "GetWeatherTool", "GetDisasterInfoTool", "AreaCalculation", "PathCalculation", "QuantityCalculation", "models"]
