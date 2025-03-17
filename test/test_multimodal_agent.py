import os
import sys
import base64
import logging

# 添加项目根目录到路径，确保能够正确导入应用模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.multimodal_agent import MultiModalAgent

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

def test_multimodal_agent():
    """测试MultiModalAgent类的基本功能"""
    # 创建MultiModalAgent实例
    agent = MultiModalAgent()  # 使用支持vision的模型
    
    # 修改测试图像路径为相对于测试文件夹的路径
    test_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(test_dir, "test_image.png")
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        logger.error(f"找不到测试图像: {image_path}")
        logger.info("请在test文件夹中放置一张名为test_image.png的图片，或修改测试脚本中的图像路径")
        return
    
    # 加载图像为base64字符串
    image_data = load_image_as_base64(image_path)
    if not image_data:
        return
    
    # 准备包含图像和文本的测试消息
    test_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "这张图片里有什么？请详细描述一下。返回信息使用中文"
                },
                {
                    "type": "image",
                    "image_data": image_data
                }
            ]
        }
    ]
    
    # 测试非流式响应
    logger.info("测试非流式响应...")
    try:
        response = agent.run(test_messages)
        logger.info(f"非流式响应内容:\n{response['message']['content']}")
    except Exception as e:
        logger.error(f"非流式测试失败: {str(e)}")
    
    # 测试流式响应
    logger.info("\n测试流式响应...")
    try:
        full_response = ""
        for chunk in agent.run_stream(test_messages):
            # 只打印部分流式块以避免日志过多
            if len(full_response) == 0:
                logger.info(f"收到第一个流式响应块: {chunk['delta']}")
            full_response += chunk["delta"]
        
        logger.info(f"完整流式响应长度: {len(full_response)} 字符")
        # 显示前200个字符的响应预览
        preview = full_response[:200] + ("..." if len(full_response) > 200 else "")
        logger.info(f"流式响应预览:\n{preview}")
    except Exception as e:
        logger.error(f"流式测试失败: {str(e)}")

def test_text_only():
    """测试纯文本输入的处理"""
    agent = MultiModalAgent()
    
    # 准备纯文本测试消息
    text_messages = [
        {
            "role": "user",
            "content": "你好，请问你是什么模型？"
        }
    ]
    
    logger.info("测试纯文本输入...")
    try:
        response = agent.run(text_messages)
        logger.info(f"纯文本响应内容:\n{response['message']['content']}")
    except Exception as e:
        logger.error(f"纯文本测试失败: {str(e)}")

if __name__ == "__main__":
    logger.info("开始测试MultiModalAgent...")
    test_multimodal_agent()
    test_text_only()
    logger.info("测试完成。")