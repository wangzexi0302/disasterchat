from app.tools.base import Tool
from typing import Dict, Any
import datetime

class GetDisasterInfoTool(Tool):
    """获取灾害信息的工具"""
    
    @property
    def name(self) -> str:
        return "get_disaster_info"
    
    @property
    def description(self) -> str:
        return "获取指定灾害类型和地区的灾害信息"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "disaster_type": {
                "type": "string",
                "description": "灾害类型，例如：地震、洪水、台风、山火",
                "enum": ["地震", "洪水", "台风", "山火", "泥石流", "干旱"],
                "required": True
            },
            "location": {
                "type": "string",
                "description": "地区名称，例如：四川、广东",
                "required": True
            }
        }
    
    def execute(self, disaster_type: str, location: str) -> str:
        """模拟获取灾害信息"""
        # 实际项目中应该查询真实的灾害数据库或API
        today = datetime.datetime.now().strftime("%Y年%m月%d日")
        
        disaster_info = {
            "地震": {
                "四川": f"{today}，四川地区发生5.2级地震，震源深度10千米，目前已安全转移人口5000人，无人员伤亡报告。",
                "云南": f"{today}，云南地区发生4.5级地震，震源深度8千米，暂无人员伤亡报告。"
            },
            "洪水": {
                "广东": f"{today}，广东珠江流域洪水预警，已安排防洪预案，请密切关注当地政府通知。",
                "湖南": f"{today}，湖南洞庭湖水位上涨，已启动防汛Ⅲ级响应。"
            },
            "台风": {
                "广东": f"{today}，'海葵'台风即将登陆广东沿海，预计风力达10-12级，请做好防范准备。",
                "浙江": f"{today}，台风'杜苏芮'影响浙江沿海，已发布台风橙色预警。"
            }
        }
        
        if disaster_type in disaster_info and location in disaster_info[disaster_type]:
            return disaster_info[disaster_type][location]
        else:
            return f"暂无{location}地区{disaster_type}灾害的相关信息。"
