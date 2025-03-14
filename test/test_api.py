import requests
import json
import time

# API 基础 URL
BASE_URL = "http://localhost:8000"

def test_chat_endpoint():
    """测试聊天端点"""
    url = f"{BASE_URL}/chat"
    
    # 请求数据
    payload = {
        "messages": [
            {"role": "user", "content": "你好，请简要介绍下自然灾害应急管理"}
        ],
        "model": "qwen2.5"  # 使用README中提到的中文模型
    }
    
    # 发送POST请求
    print(f"发送请求到 {url}...")
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    
    # 输出结果
    print(f"状态码: {response.status_code}")
    print(f"响应时间: {end_time - start_time:.2f} 秒")
    
    if response.status_code == 200:
        print("成功! 响应内容:")
        try:
            # 尝试格式化JSON输出
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        except:
            # 如果不是JSON，直接打印文本
            print(response.text)
    else:
        print(f"错误! 响应内容: {response.text}")
    
    return response.status_code == 200

def test_health_endpoint():
    """测试健康检查端点"""
    url = f"{BASE_URL}/health"
    
    print(f"发送请求到 {url}...")
    response = requests.get(url)
    
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        print("成功! 响应内容:")
        try:
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        except:
            print(response.text)
    else:
        print(f"错误! 响应内容: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    # 运行测试
    print("=== 测试健康检查端点 ===")
    health_ok = test_health_endpoint()
    
    print("\n=== 测试聊天端点 ===")
    chat_ok = test_chat_endpoint()
    
    # 汇总结果
    print("\n=== 测试结果汇总 ===")
    print(f"健康检查端点: {'通过' if health_ok else '失败'}")
    print(f"聊天端点: {'通过' if chat_ok else '失败'}")
    
    # 根据测试结果设置退出码
    if health_ok and chat_ok:
        print("\n所有测试通过 ✓")
    else:
        print("\n测试失败 ✗")
