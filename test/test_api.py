import asyncio
import httpx
import json

# 定义 API 端点
CHAT_SEND_URL = "http://localhost:8000/api/chat/send"
HISTORY_DETAIL_URL = "http://localhost:8000/api/chat/history_detail"

# 模拟发送消息的请求数据
send_request_data = {
    "sessionId": "f4c21ed1-857d-4b3b-bab7-ecf0c4b06c81",
    "message": {
        "role": "user",
        "content": "请提供具体的灾后救援建议和资源分配方案"
    }
}

# 模拟获取历史详情的请求数据
history_request_data = {
    "sessionId": "6623ac9d-454f-4eb3-9f63-31814d4e2fc6"
}


async def send_chat_message(client):
    """
    异步发送聊天消息并处理流式响应
    :param client: httpx 异步客户端
    """
    async with client.stream("POST", CHAT_SEND_URL, json=send_request_data) as response:
        async for line in response.aiter_lines():
            if line:
                try:
                    data = json.loads(line.replace("data: ", ""))
                    print(f"Received chat message chunk: {data}")
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON: {line}")


async def get_history_detail(client):
    """
    异步获取历史会话详情
    :param client: httpx 异步客户端
    """
    response = await client.post(HISTORY_DETAIL_URL, json=history_request_data)
    if response.status_code == 200:
        data = response.json()
        print(f"Received history detail: {data}")
    else:
        print(f"Failed to get history detail. Status code: {response.status_code}")


async def main():
    """
    主函数，并发执行发送消息和获取历史详情的任务
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            send_chat_message(client),
            get_history_detail(client),
            get_history_detail(client),
            get_history_detail(client),
            get_history_detail(client),
            get_history_detail(client),
            get_history_detail(client),
            get_history_detail(client),
            get_history_detail(client),
            get_history_detail(client)

        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())