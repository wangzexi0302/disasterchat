from datetime import datetime
import os
import sys
import logging
from typing import Dict, List, Any

# 添加项目根目录到 Python 路径，以便导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.sentimodel_agent import SentiModelAgent

log_file = os.path.join('test/', f"app_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_sentimodel_agent():
    """测试 SentiModelAgent 类的基本功能"""
    # 创建 SentiModelAgent 实例
    agent = SentiModelAgent()
    
    # 准备测试数据，包括图像类型和测试消息
    pic_type = "disaster"  # 灾害图像类型
    
    # 准备测试消息 - 可以测试不同类型的意图
    test_messages = [
        {
            "role": "user",
            "content": "请描述灾后影像的具体信息"
        }
    ]
    
    # 另一组测试消息 - 用于测试不同的意图识别
    test_messages1 = [
        {
            "role": "user",
            "content": "什么是地质灾害，飓风又是如何产生的。"
        }
    ]
    
    # 测试 run 方法 (流式响应)
    logger.info("测试 SentiModelAgent 流式响应...")
    try:
        logger.info("测试建筑损毁评估意图:")
        full_response = ""
        for chunk in agent.run(test_messages, pic_type):
            # 只打印部分流式块以避免日志过多
            if len(full_response) == 0:
                logger.info(f"收到第一个流式响应块: {chunk}")
            full_response += chunk
        
        logger.info(f"完整流式响应长度: {len(full_response)} 字符")
        # 显示前200个字符的响应预览
        logger.info(f"流式响应预览:\n{full_response}")
        
        # 测试第二组消息
        # logger.info("\n测试灾情概览意图:")
        # full_response = ""
        # for chunk in agent.run(test_messages_2, pic_type):
        #     if len(full_response) == 0:
        #         logger.info(f"收到第一个流式响应块: {chunk}")
        #     full_response += chunk
        
        # logger.info(f"完整流式响应长度: {len(full_response)} 字符")
        # preview = full_response[:200] + ("..." if len(full_response) > 200 else "")
        # logger.info(f"流式响应预览:\n{preview}")
        
    except Exception as e:
        logger.error(f"流式测试失败: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    test_sentimodel_agent()
