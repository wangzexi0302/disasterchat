# -*- coding: utf-8 -*-
from app.tools.base import Tool
from typing import Dict, Any
from app.agents.image_analysis_agent import ImageAnalysisAgent


class CallImageAnalysis(Tool):
    """调用遥感影像分析模型"""

    @property
    def name(self) -> str:
        return "call_image_analysis"
    
    @property
    def description(self) -> str:
        return ("根据用户要求对遥感影像进行处理（变化检测、语义分割），并调用路径计算、面积计算、数量计算工具，"
                "实现道路可达性估计、各类地物受灾面积估计、受灾建筑物损伤情况统计")
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "message": {
                "type": "string",
                "description": "用户输入的图片和历史消息",
                "required": True
            }
        }
    
    def execute(self, message: str) -> str:
        agent = ImageAnalysisAgent()
        agent.run(message)
        return 

