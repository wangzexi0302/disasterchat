import requests
import json
import uuid
import os
from pydantic import BaseModel, ValidationError

# ======================
# 配置
# ======================
BASE_URL = "http://localhost:8000"  # 本地服务地址
TEST_IMAGE_PATH = "test_image.jpg"  # 测试图片路径（需提前准备）


# ======================
# 辅助函数
# ======================
def print_separator(title):
    print(f"\n{'=' * 50}")
    print(f"=== {title} ===")
    print(f"{'=' * 50}")


def validate_response(response, expected_status=200, request_url=None, request_body=None):
    """验证响应状态码和 JSON 格式，输出详细错误信息"""
    if response.status_code != expected_status:
        error_msg = f"请求 {request_url} 时状态码错误：预期 {expected_status}，实际 {response.status_code}"
        if request_body:
            error_msg += f"\n请求体: {json.dumps(request_body, ensure_ascii=False, indent=2)}"
        error_msg += f"\n响应内容：{response.text}"
        raise AssertionError(error_msg)
    try:
        return response.json()
    except json.JSONDecodeError:
        error_msg = f"请求 {request_url} 时响应不是有效的 JSON"
        if request_body:
            error_msg += f"\n请求体: {json.dumps(request_body, ensure_ascii=False, indent=2)}"
        error_msg += f"\n响应内容：{response.text}"
        raise AssertionError(error_msg)


# ======================
# 测试用例
# ======================
def test_create_session():
    """测试创建新会话"""
    print_separator("创建新会话")
    url = f"{BASE_URL}/api/chat/new_session"
    response = requests.post(url)
    data = validate_response(response, request_url=url)
    assert "session_id" in data, f"响应中缺少 session_id，响应内容：{response.text}"
    print(f"创建会话成功，ID：{data['session_id']}")
    return data["session_id"]


def test_upload_image(session_id):
    """测试上传图片"""
    print_separator("上传图片")
    url = f"{BASE_URL}/api/chat/upload_image"

    # 准备测试图片（需存在）
    if not os.path.exists(TEST_IMAGE_PATH):
        with open(TEST_IMAGE_PATH, "wb") as f:
            f.write(b"test image content")  # 创建临时测试图片

    with open(TEST_IMAGE_PATH, "rb") as f:
        files = {"files": (TEST_IMAGE_PATH, f, "image/jpeg")}
        response = requests.post(url, files=files)
    data = validate_response(response, request_url=url)
    assert data["status"] == "success", f"图片上传失败，响应内容：{response.text}"
    assert len(data["image_ids"]) == 1, f"图片 ID 数量错误，响应内容：{response.text}"
    print(f"上传成功，图片 ID：{data['image_ids'][0]}")
    return data["image_ids"][0]


def test_send_text_message(session_id):
    """测试发送纯文本消息"""
    print_separator("发送纯文本消息")
    url = f"{BASE_URL}/api/chat/send"
    message = {
        "role": "user",
        "content": "你好，这是纯文本消息"
    }
    payload = {
        "sessionId": session_id,
        "message": message
    }
    response = requests.post(url, json=payload)
    validate_response(response, expected_status=200, request_url=url, request_body=payload)

    # 解析流式响应（简化处理，实际需处理 event-stream 格式）
    print("流式响应示例（前 50 字符）：")
    print(response.text[:50] + "...")


def test_send_multimodal_message(session_id, image_id):
    """测试发送多模态（图文）消息"""
    print_separator("发送多模态消息")
    url = f"{BASE_URL}/api/chat/send"
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "这是一张图片："},
            {"type": "image", "image_id": image_id}
        ]
    }
    payload = {
        "sessionId": session_id,
        "message": message
    }
    response = requests.post(url, json=payload)
    validate_response(response, expected_status=200, request_url=url, request_body=payload)
    print("多模态消息发送成功（流式响应已简化）")


def test_get_history(session_id):
    """测试获取历史消息"""
    print_separator("获取历史消息")
    url = f"{BASE_URL}/api/chat/history_detail"
    payload = {
        "sessionId": session_id
    }
    response = requests.post(url, json=payload)
    data = validate_response(response, request_url=url, request_body=payload)
    messages = data["data"]["messages"]
    assert len(messages) >= 1, f"历史消息为空，响应内容：{response.text}"
    print(f"获取到 {len(messages)} 条消息，最近一条：")
    print(json.dumps(messages[-1], ensure_ascii=False, indent=2))


def test_prompts_template():
    """测试预设提示"""
    print_separator("获取预设提示")
    url = f"{BASE_URL}/api/prompts/templates"
    response = requests.post(url)
    data = validate_response(response, request_url=url)
    templates = data["data"]["templates"]
    assert len(templates) >= 1, f"预设提示为空，响应内容：{response.text}"
    print(f"获取到 {len(templates)} 个预设提示：")
    print([t["name"] for t in templates])


# ======================
# 执行测试
# ======================
if __name__ == "__main__":
    try:
        # 1. 创建会话
        session_id = test_create_session()

        # 2. 上传图片（可选）
        image_id = test_upload_image(session_id) if os.path.exists(TEST_IMAGE_PATH) else None

        # 3. 发送纯文本消息
        test_send_text_message(session_id)

        # 4. 发送多模态消息（需图片上传成功）
        if image_id:
            test_send_multimodal_message(session_id, image_id)

        # 5. 获取历史消息
        test_get_history(session_id)

        # 6. 测试预设提示
        test_prompts_template()

        print("\n所有测试通过 ✅")

    except AssertionError as e:
        print(f"\n测试失败 ❌：{str(e)}")
    except Exception as e:
        print(f"\n发生异常 ❌：{str(e)}")
