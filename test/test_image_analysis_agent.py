import os
import sys
import base64
import logging

# 添加项目根目录到路径，确保能够正确导入应用模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.image_analysis_agent import ImageAnalysisAgent

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


def test_image_analysis_agent(sample_index, points):
    """测试ImageAnalysisAgent类的基本功能"""
    # 创建MultiModalAgent实例
    agent = ImageAnalysisAgent()

    # 加载图像为base64字符串
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_pre_path = os.path.join(root_path, f"./test/assests/{sample_index}/pre.png")
    image_post_path = os.path.join(root_path, f"./test/assests/{sample_index}/post.png")

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
                    "text": "【测试内容】"
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
    logger.info("测试【无工具】的非流式响应...")
    test_messages[0]['content'][0]['text'] = "请问你是哪种大语言模型？"
    try:
        response = agent.run(test_messages, pic_type='both', sample_index=sample_index)
        logger.info(f"非流式响应内容:\n{response['content'][0]}")
    except Exception as e:
        logger.error(f"非流式测试失败: {str(e)}")

    logger.info("测试【路径计算工具：评估A到B是否可达】的非流式响应...")
    test_messages[0]['content'][0]['text'] = f"请根据我给你的**受灾前**的遥感影像，评估**灾前**起始点A到终止点B是否可通行。{points}。返回信息使用中文。"
    try:
        response = agent.run(test_messages, pic_type='both', sample_index=sample_index)
        logger.info(f"非流式响应内容:\n{response['content'][0]}")
    except Exception as e:
        logger.error(f"非流式测试失败: {str(e)}")

    logger.info("测试【路径计算工具：评估A到B是否可达】的非流式响应...")
    test_messages[0]['content'][0]['text'] = f"请根据我给你的**受灾后**的遥感影像，评估**灾后**起始点A到终止点B是否可通行。{points}。返回信息使用中文。"
    try:
        response = agent.run(test_messages, pic_type='both', sample_index=sample_index)
        logger.info(f"非流式响应内容:\n{response['content'][0]}")
    except Exception as e:
        logger.error(f"非流式测试失败: {str(e)}")

    logger.info("测试【数量计算工具：建筑物损伤统计】的非流式响应...")
    test_messages[0]['content'][0]['text'] = "请根据我给你的受灾前后的遥感影像，分析建筑物的受损严重程度。返回信息使用中文。"
    try:
        response = agent.run(test_messages, pic_type='both', sample_index=sample_index)
        logger.info(f"非流式响应内容:\n{response['content'][0]}")
    except Exception as e:
        logger.error(f"非流式测试失败: {str(e)}")

    logger.info("测试【面积计算工具：各类地物受灾面积统计】的非流式响应...")
    test_messages[0]['content'][0]['text'] = "请根据我给你的受灾前后的遥感影像，统计建筑物、道路等各类地物的受灾面积。返回信息使用中文。"
    try:
        response = agent.run(test_messages, pic_type='both', sample_index=sample_index)
        logger.info(f"非流式响应内容:\n{response['content'][0]}")
    except Exception as e:
        logger.error(f"非流式测试失败: {str(e)}")

    logger.info("测试【多种计算工具】的非流式响应...")
    test_messages[0]['content'][0]['text'] = f"请根据我给你的受灾前后的遥感影像，完成以下3项任务：\n1. 评估建筑物的受损严重程度；\n2. 统计各类地物的受灾面积；\n3. 评估受灾后从A到B道路是否可通行。{points}。返回信息使用中文。"
    try:
        response = agent.run(test_messages, pic_type='both', sample_index=sample_index)
        logger.info(f"非流式响应内容:\n{response['content'][0]}")
    except Exception as e:
        logger.error(f"非流式测试失败: {str(e)}")

    # 测试流式响应
    logger.info("测试【无工具】的流式响应...")
    test_messages[0]['content'][0]['text'] = "请问你是哪种大语言模型？"
    try:
        full_response = ""
        for chunk in agent.run_stream(test_messages, pic_type='both', sample_index=sample_index):
            # 只打印部分流式块以避免日志过多
            if not chunk['done']:
                if len(full_response) == 0:
                    logger.info(f"流式响应块文本: {chunk['delta']}")
                full_response += chunk["delta"]
            else:
                logger.info("响应完成")
        
        logger.info(f"完整流式响应长度: {len(full_response)} 字符")
        # 显示前200个字符的响应预览
        preview = full_response[:200] + ("..." if len(full_response) > 200 else "")
        logger.info(f"流式响应预览:\n{preview}")
    except Exception as e:
        logger.error(f"流式测试失败: {str(e)}")

    logger.info("测试【多种计算工具】的流式响应...")
    test_messages[0]['content'][0]['text'] = f"请根据我给你的受灾前后的遥感影像，完成以下3项任务：\n1. 评估建筑物的受损严重程度；\n2. 统计各类地物的受灾面积；\n3. 评估受灾后从A到B道路是否可通行。{points}。返回信息使用中文。"
    try:
        full_response = ""
        for chunk in agent.run_stream(test_messages, pic_type='both', sample_index=sample_index):
            # 只打印部分流式块以避免日志过多
            if not chunk['done']:
                if len(full_response) == 0:
                    logger.info(f"流式响应块文本: {chunk['delta']}")
                full_response += chunk["delta"]
            else:
                logger.info("响应完成")

        logger.info(f"完整流式响应长度: {len(full_response)} 字符")
        # 显示前200个字符的响应预览
        preview = full_response[:200] + ("..." if len(full_response) > 200 else "")
        logger.info(f"流式响应预览:\n{preview}")
    except Exception as e:
        logger.error(f"流式测试失败: {str(e)}")


if __name__ == "__main__":
    logger.info("开始测试...")
    for sample_index in range(1, 5):
        logger.info(f"开始测试第 {sample_index} 个样本...")
        points = "{\"pre\": [{\"x\": 630,\"y\": 650},{\"x\": 352,\"y\": 372}],\"post\":[{\"x\": 468,\"y\":488}, {\"x\": 706,\"y\": 726}]}"
        if sample_index == 1:
            points = "{\"pre\": [{\"x\": 1000,\"y\": 600},{\"x\": 920,\"y\": 700}],\"post\":[{\"x\": 1000,\"y\":600}, {\"x\": 920,\"y\": 700}]}"
        elif sample_index == 2:
            points = "{\"pre\": [{\"x\": 1000,\"y\": 500},{\"x\": 500,\"y\": 900}],\"post\":[{\"x\": 1000,\"y\":500}, {\"x\": 500,\"y\": 900}]}"
        elif sample_index == 3:
            points = "{\"pre\": [{\"x\": 100,\"y\": 50},{\"x\": 600,\"y\": 50}],\"post\":[{\"x\": 100,\"y\":50}, {\"x\": 600,\"y\": 50}]}"
        elif sample_index == 4:
            points = "{\"pre\": [{\"x\": 250,\"y\": 25},{\"x\": 850,\"y\": 200}],\"post\":[{\"x\": 250,\"y\":25}, {\"x\": 850,\"y\": 200}]}"
        test_image_analysis_agent(sample_index, points)
    logger.info("测试完成。")