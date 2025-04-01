from app.tools.base import Tool
from typing import Dict, Any
from app.agents.qa_agent import QAAgent


class CallQAAgent(Tool):
    """调用QAAgent"""

    @property
    def name(self) -> str:
        return "call_qaagent"
    
    @property
    def description(self) -> str:
        return "根据用户输入回答地灾方面的专业知识"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "message": {
                "type": "string",
                "description": "用户历史消息总结",
                "required": True
            },
        }
    
    
    
    def execute(self, message: str, pic_type: str) -> str:
        agent = QAAgent()      
        return agent.run(message, pic_type)