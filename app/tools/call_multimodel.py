from app.tools.base import Tool
from typing import Dict, Any
from app.agents import MultiModalAgent
class CallMultiModel(Tool):
    """调用多模态模型"""

    @property
    def name(self) -> str:
        return "call_multimodel"
    
    @property
    def description(self) -> str:
        return "根据用户要求对图片内容进行分析"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "message": {
                "type": "string",
                "description": "用户输入的图片和历史消息",
                "required": True
            },
        }
    
    
    
    def execute(self, message: str) -> str:
        agent = MultiModalAgent()
        agent.run(message)
        return 

