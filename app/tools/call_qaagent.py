from app.tools.base import Tool
from typing import Dict, Any
import logging
from app.agents.qa_agent import QAAgent

logger = logging.getLogger(__name__)

class CallQAAgent(Tool):
    """调用QAAgent回答问题"""

    @property
    def name(self) -> str:
        return "call_qaagent"
    
    @property
    def description(self) -> str:
        return "根据用户输入回答地灾相关的专业问题或一般性问题"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "message": {
                "type": "string",
                "description": "用户的问题或查询内容",
                "required": True
            }   
        }
    
    def execute(self, message: str) -> str:
        """
        执行问答功能
        
        Args:
            message: 用户输入的问题或查询内容
            
        Returns:
            QAAgent的回答结果
        """
        logger.info(f"调用QAAgent回答问题: {message[:50]}...")
        
        agent = QAAgent()
        try:
            result = agent.run(message)
            logger.info("QAAgent回答完成")
            return result
        except Exception as e:
            logger.error(f"QAAgent回答失败: {str(e)}", exc_info=True)
            return f"问题回答失败: {str(e)}"