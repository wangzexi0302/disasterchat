import requests
import json
import uuid
import os
from pydantic import BaseModel, ValidationError

# ======================
# 配置
# ======================
BASE_URL = "http://localhost:8000"  # 本地服务地址
TEST_IMAGE_PATH = r"C:\Users\11758\Desktop\disasterchat\test\test_image.png"  # 测试图片路径（需提前准备）


# ======================
# 辅助函数
# ======================
def print_separator(title):
    print(f"\n{'=' * 50}")
    print(f"=== {title} ===")
    print(f"{'=' * 50}")


def validate_response(response, expected_status=200, request_url=None, request_body=None):
    """验证响应状态码和 JSON 格式，输出详细错误信息"""
    print(f"\n📥 响应状态码: {response.status_code}")
    print(f"📥 响应内容: {response.text}")  # 新增：打印响应内容

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


def validate_response_stream(response, expected_status=200, request_url=None, request_body=None):
    """验证流式响应的每个 JSON 块"""
    print(f"\n📥 响应状态码: {response.status_code}")
    print(f"📥 流式响应内容:")  # 新增：打印流式响应内容

    if response.status_code != expected_status:
        error_msg = f"请求 {request_url} 时状态码错误：预期 {expected_status}，实际 {response.status_code}"
        if request_body:
            error_msg += f"\n请求体: {json.dumps(request_body, ensure_ascii=False, indent=2)}"
        error_msg += f"\n响应内容：{response.text}"
        raise AssertionError(error_msg)
    
    final_chunk = None
    for line in response.iter_lines():
        if not line:
            continue
        
        line = line.decode().strip()
        print(f"  → {line}")  # 新增：打印每个响应块
        
        # 跳过非数据行（如 event: ping）
        if line.startswith("event:"):
            continue
        
        # 提取数据部分（格式：data: {"key": "value"}）
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        
        # 替换单引号为双引号
        line = line.replace("'", "\"")
        
        try:
            chunk = json.loads(line)
            assert "data" in chunk, f"响应块缺少 'data' 字段：{line}"
            if chunk["data"].get("done"):
                final_chunk = chunk
        except json.JSONDecodeError as e:
            error_msg = f"解析 JSON 块失败：{line}"
            if request_body:
                error_msg += f"\n请求体: {json.dumps(request_body, ensure_ascii=False, indent=2)}"
            raise AssertionError(error_msg) from e
    
    assert final_chunk is not None, "未收到流式响应的结束标记"
    assert final_chunk["data"]["done"] is True, "流式响应未正确结束"


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
        data = {"type": "pre"}
        response = requests.post(url,data=data,files=files)
    data = validate_response(response, request_url=url)
    assert data["status"] == "success", f"图片上传失败，响应内容：{response.text}"
    assert len(data["image_ids"]) == 1, f"图片 ID 数量错误，响应内容：{response.text}"
    print(f"上传成功，图片 ID:{data['image_ids'][0]}")
    return data["image_ids"][0]


def test_send_text_message(session_id):
    """测试发送纯文本消息"""
    print_separator("发送纯文本消息")
    url = f"{BASE_URL}/api/chat/send"
    message = {
        "role": "user",
        "content":[{"type": "text", "text": "你好你好你糊弄很好奇哈哈哈哈哈哈哈"}]  

    }
    payload = {
        "sessionId": session_id,
        "message": message
    }
    
    response = requests.post(url, json=payload)
    validate_response_stream(response, expected_status=200, request_url=url, request_body=payload)
    print("流式响应验证成功")

def test_send_text_message2(session_id):
    """测试发送纯文本消息"""
    print_separator("发送纯文本消息")
    url = f"{BASE_URL}/api/chat/send"
    message = {
        "role": "user",
        "content": "你好，你是什么东西"
    }
    payload = {
        "sessionId": session_id,
        "message": message
    }
    
    response = requests.post(url, json=payload)
    validate_response_stream(response, expected_status=200, request_url=url, request_body=payload)
    print("流式响应验证成功")



def test_send_multimodal_message(session_id, image_id):
    """测试发送多模态（图文）消息"""
    print_separator("发送多模态消息")
    url = f"{BASE_URL}/api/chat/send"
    
    # 正确的请求体结构
    payload = {
        "sessionId": session_id,  # 会话 ID 放在顶层
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是一张图片："},  # 文本内容
                {"type": "image", "image_id": image_id}  # 图片内容（仅需 image_id）
            ]
        }
    }
    
    response = requests.post(url, json=payload)
    validate_response_stream(response, expected_status=200, request_url=url, request_body=payload)
    print("多模态流式响应验证成功")


def test_get_history_list():
    """测试获取历史会话列表"""
    print_separator("获取历史会话列表")
    url = f"{BASE_URL}/api/chat/history_list"
    request_body = {}  # 无参数

    # 发送请求
    response = requests.post(url, json=request_body)
    data = validate_response(response, request_url=url, request_body=request_body)

    # 断言响应结构
    assert "data" in data, "响应中缺少 'data' 字段"
    assert "templates" in data["data"], "响应中缺少 'templates' 字段"
    
    templates = data["data"]["templates"]
    assert isinstance(templates, list), "'templates' 不是列表"
    
    if templates:  # 有数据时验证字段
        first_session = templates[0]
        assert "sessionId" in first_session, "会话缺少 'id' 字段"
        assert "name" in first_session, "会话缺少 'created_at' 字段"
    
    print(f"成功获取 {len(templates)} 个历史会话")

def test_get_history_detail_exists(session_id):
    """测试存在的会话ID"""
    print_separator("获取历史消息（存在的会话）")
    url = f"{BASE_URL}/api/chat/history_detail"
    request_body = {"sessionId": session_id}  # 注意字段名是 sessionId

    # 发送请求
    response = requests.post(url, json=request_body)
    data = validate_response(response, request_url=url, request_body=request_body)

    # 断言响应结构
    assert "data" in data, "响应中缺少 'data' 字段"
    assert "messages" in data["data"], "响应中缺少 'messages' 字段"
    
    messages = data["data"]["messages"]
    assert len(messages) >= 1, "历史消息为空"
    
    # 断言消息内容（示例：检查第一条消息）
    first_msg = messages[0]
    assert "role" in first_msg, "消息缺少 'role' 字段"
    assert "content" in first_msg, "消息缺少 'content' 字段"
    assert "images" in first_msg, "消息缺少 'images' 字段"
    
    print(f"成功获取 {len(messages)} 条消息")



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


def test_get_session_title_summary(session_id):
    """测试获取会话标题总结"""
    print_separator("获取会话标题总结")
    
    # 测试存在用户和AI消息的情况
    print_separator("测试存在用户和AI消息")
    url = f"{BASE_URL}/api/session/{session_id}/title_summary"
    response = requests.get(url)
    data = validate_response(response, expected_status=200, request_url=url)
    assert len(data["summary"]) <= 10, f"总结超过10字：{data['summary']}"
    print(f"组合总结结果：{data['summary']}")

# ======================
# 执行测试
# ======================
if __name__ == "__main__":
    try:
        # 1. 创建会话
        #session_id = test_create_session()

        # 2. 上传图片（可选）
        #image_id = test_upload_image(session_id) if os.path.exists(TEST_IMAGE_PATH) else None

        # 3. 发送纯文本消息
        test_send_text_message("6623ac9d-454f-4eb3-9f63-31814d4e2fc6")


        # 4. 发送多模态消息（需图片上传成功）
        #if image_id:
            #test_send_multimodal_message(session_id, image_id)

        # 5. 获取历史消息
        #test_get_history_list()

        #6. 测试获取历史消息（存在的会话）
        #test_get_history_detail_exists("3bad6f94-1bc3-43f7-aaca-8155c0ca8586")

        # 6. 测试预设提示
        #test_prompts_template()

        # 7 测试获取会话标题总结
        #test_get_session_title_summary("111756cc-7a5f-44c0-a079-569cc5b18d4a")

        #print("\n所有测试通过 ✅")

    except AssertionError as e:
        print(f"\n测试失败 ❌：{str(e)}")
    except Exception as e:
        print(f"\n发生异常 ❌：{str(e)}")