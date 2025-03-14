from app.tools.base import Tool
from typing import Dict, Any
import requests

class GetWeatherTool(Tool):
    """获取天气情况的工具"""
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "获取指定城市的天气信息"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "city": {
                "type": "string",
                "description": "城市名称，例如：北京、上海",
                "required": True
            }
        }
    
    def execute(self, city: str) -> str:
        """模拟获取天气信息"""
        # 实际项目中应该调用真实的天气API
        # 这里返回模拟数据
        weather_data = {
            "北京": "晴朗，温度23℃，湿度45%，风力2级",
            "上海": "多云，温度26℃，湿度60%，风力3级",
            "广州": "小雨，温度28℃，湿度80%，风力1级",
        }
        
        return weather_data.get(city, f"无法获取{city}的天气信息，请检查城市名称是否正确")
