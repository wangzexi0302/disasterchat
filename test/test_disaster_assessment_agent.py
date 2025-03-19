import os
import sys
import base64
import logging

# 添加项目根目录到路径，确保能够正确导入应用模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.disaster_impact_assessment_agent import DisasterImpactAssessmentAgent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image_as_base64(image_path):
    """将图像文件加载为base64编码字符串"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        logger.error(f"加载图像失败: {str(e)}")
        return None


def test_impact_assessment_agent():
    """测试ImpactAssessmentAgent类的基本功能"""
    # 创建MultiModalAgent实例
    agent = DisasterImpactAssessmentAgent()

    # 加载图像为base64字符串
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_pre_path = os.path.join(root_path, "./demo_data/hurricane-florence_00000018_pre_disaster.png")
    image_post_path = os.path.join(root_path, "./demo_data/hurricane-florence_00000018_post_disaster.png")

    image_pre_data = load_image_as_base64(image_pre_path)
    image_post_data = load_image_as_base64(image_post_path)
    if not image_pre_data or not image_post_data:
        return
    
    # 准备包含图像和文本的测试消息
    test_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请分析该地区的受灾情况，评估受灾范围。返回信息使用中文"
                },
                {
                    "type": "image",
                    "image_data": image_pre_data
                },
                {
                    "type": "image",
                    "image_data": image_post_data
                }
            ]
        }
    ]
    
    # 测试非流式响应
    logger.info("测试非流式响应...")
    try:
        response = agent.run(test_messages)
        logger.info(f"非流式响应内容: {response['message']['content']}")
        logger.info(f"非流式响应图片: {len(response['message']['image'])}")
    except Exception as e:
        logger.error(f"非流式测试失败: {str(e)}")
    
    # 测试流式响应
    logger.info("\n测试流式响应...")
    try:
        full_response = ""
        for chunk in agent.run_stream(test_messages):
            # 只打印部分流式块以避免日志过多
            if not chunk['done']:
                if len(full_response) == 0:
                    logger.info(f"流式响应块文本: {chunk['delta']}")
                full_response += chunk["delta"]
            else:
                logger.info(f"流式响应块图片: {len(chunk['image'])}")
                logger.info("响应完成")
        
        logger.info(f"完整流式响应长度: {len(full_response)} 字符")
        # 显示前200个字符的响应预览
        preview = full_response[:200] + ("..." if len(full_response) > 200 else "")
        logger.info(f"流式响应预览:\n{preview}")
    except Exception as e:
        logger.error(f"流式测试失败: {str(e)}")


if __name__ == "__main__":
    logger.info("开始测试ImpactAssessmentAgent...")
    test_impact_assessment_agent()
    logger.info("测试完成。")