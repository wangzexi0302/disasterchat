# 此文件用于标记工具目录为Python包
from app.tools.base import Tool
from app.tools.weather import GetWeatherTool
from app.tools.disaster_info import GetDisasterInfoTool
from app.tools.call_multimodel import CallMultiModel

# 导出所有可用工具
available_tools = [
    GetWeatherTool(),
    GetDisasterInfoTool()
]

models = [
    CallMultiModel()
]

__all__ = ["Tool", "available_tools", "GetWeatherTool", "GetDisasterInfoTool", "models"]
