from app.tools.base import Tool
from typing import Dict, Any, Literal
import logging
from app.agents.multimodel_agent_new import MultiModalAgent

logger = logging.getLogger(__name__)

class CallMultiModel(Tool):
    """调用多模态模型分析灾害影像"""

    @property
    def name(self) -> str:
        return "call_multimodel"
    
    @property
    def description(self) -> str:
        return "根据用户要求对灾害影像进行粗略分析，可以分析灾前影像、灾后影像或同时分析两种影像"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "message": {
                "type": "string",
                "description": "用户的具体问题或分析请求",
                "required": True
            },
            "pic_type": {
                "type": "string",
                "description": "要分析的影像类型：'pre'代表灾害前影像，'post'代表灾害后影像，'both'代表同时分析两种影像",
                "enum": ["pre", "post", "both"],
                "required": True
            }
        }
    
    def execute(self, message: str, pic_type: str, **kwargs) -> str:
        """
        执行多模态模型分析
        
        Args:
            message: 用户输入的问题或分析请求
            pic_type: 影像类型，可选值为 "pre"(灾前), "post"(灾后) 或 "both"(两种都分析)
            
        Returns:
            模型分析结果
        """
        # 验证pic_type参数
        if pic_type not in ["pre", "post", "both"]:
            logger.warning(f"无效的pic_type值: {pic_type}，将使用默认值'pre'")
            pic_type = "pre"
            
        logger.info(f"调用多模态模型分析{pic_type}类型影像，消息: {message[:50]}...")
        
        agent = MultiModalAgent()
        try:
            result = agent.run(message, pic_type)
            logger.info("多模态模型分析完成")
            return result
        except Exception as e:
            logger.error(f"多模态模型分析失败: {str(e)}", exc_info=True)
            return f"影像分析失败: {str(e)}"

