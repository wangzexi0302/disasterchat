# 此文件用于标记工具目录为Python包
from app.tools.base import Tool
from app.tools.weather import GetWeatherTool
from app.tools.disaster_info import GetDisasterInfoTool
from app.tools.call_multimodel import CallMultiModel
from app.tools.call_qaagent import CallQAAgent
# from app.tools.call_image_analysis import CallImageAnalysis


# 导出所有可用工具
available_tools = [
    GetWeatherTool(),
    GetDisasterInfoTool(),
    ]

models = [
    CallMultiModel(),
    CallQAAgent()
    # CallImageAnalysis()
]

__all__ = ["Tool", "available_tools", "GetWeatherTool", "GetDisasterInfoTool", "models"]
