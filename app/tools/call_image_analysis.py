# -*- coding: utf-8 -*-
from app.tools.base import Tool
from typing import Dict, Any, List
import logging
from app.agents.image_analysis_agent import ImageAnalysisAgent

logger = logging.getLogger(__name__)

class CallImageAnalysis(Tool):
    """调用图像分析功能执行详细分析"""

    @property
    def name(self) -> str:
        return "call_image_analysis_agent"
    
    @property
    def description(self) -> str:
        return ("根据用户要求对遥感影像进行处理（变化检测、语义分割），并调用路径计算、面积计算、数量计算工具，实现道路可达性估计、各类地物受灾面积估计、受灾建筑物损伤情况统计")
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "message": {
                "type": "string",
                "description": "用户的具体分析请求，包括要分析的内容和关注点",
                "required": True
            },
            "pic_type": {
                "type": "string",
                "description": "要分析的影像类型：'pre'代表灾害前影像，'post'代表灾害后影像，'both'代表需要同时分析两种影像(变化检测)",
                "enum": ["pre", "post", "both"],
                "required": True
            }
        }
    
    def execute(self, message: str, pic_type: str, sample_index: int = 0, **kwargs) -> str:
        """
        执行详细的灾害影像分析
        Args:
            message: 用户的具体分析请求
            pic_type: 影像类型，可选值为 "pre"(灾前), "post"(灾后) 或 "both"(变化检测)
            
        Returns:
            图像分析结果
        """
        # 验证pic_type参数
        if pic_type not in ["pre", "post", "both"]:
            logger.warning(f"无效的pic_type值: {pic_type}，将使用默认值'both'")
            pic_type = "both"
            
        logger.info(f"调用图像分析功能分析{pic_type}类型影像，分析请求: {message[:50]}...")
    
        agent = ImageAnalysisAgent()
        try:
            result = agent.run([{
                "role": 'user',
                "content": message
            }], pic_type, sample_index)
            logger.info(f"图像分析完成{result}")
            return result
        except Exception as e:
            logger.error(f"图像分析失败: {str(e)}", exc_info=True)
            return f"详细图像分析失败: {str(e)}"

