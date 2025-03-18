from fastapi import APIRouter, Request, HTTPException,BackgroundTasks,Depends
from fastapi.responses import StreamingResponse
import uuid
import time
import redis
import json
import asyncio
from pydantic import BaseModel
from app.api.database.db_setup import get_db  # 从数据库配置文件中导入获取数据库会话的函数
from app.api.database.models import DBSession  # 从数据库模型文件中导入会话模型
from sqlalchemy.orm import Session

# 创建路由
router = APIRouter(prefix="/api", tags=["chat"])

# 连接 Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)


# 定义请求和响应模型

# 获取历史对话列表响应模型
class HistoryListResponse(BaseModel):
    data: dict


# 获取历史详细对话响应模型
class HistoryDetailResponse(BaseModel):
    data: dict


# 上传图片请求模型
class UploadImageRequest(BaseModel):
    image_list: list


# 上传图片响应模型
class UploadImageResponse(BaseModel):
    status: str
    message: str = None


# 开启新会话响应模型
class NewSessionResponse(BaseModel):
    status: str
    session_id: str


# 发送消息请求模型
class SendMessageRequest(BaseModel):
    sessionId: int
    message: str


# 预设提示响应模型
class PromptsResponse(BaseModel):
    data: dict


# 4. 开启新会话
@router.post("/chat/new_session", response_model=NewSessionResponse)
async def new_session(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())
    session_number = redis_client.incr("session_counter")
    session_name = f"会话 {session_number}"
    session_key = f"session:{session_id}"

    redis_client.hset(session_key, "sessionId", session_id)
    redis_client.hset(session_key, "name", session_name)

    def save_session_to_mysql():
        try:
            new_session = DBSession(id=session_id, name=session_name)
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
        except Exception as e:
            db.rollback()
            print(f"保存会话到 MySQL 时出错: {e}")
        finally:
            db.close()

    background_tasks.add_task(save_session_to_mysql)

    return NewSessionResponse(status="success", session_id=session_id)

# 1. 获取历史对话列表
@router.post("/chat/history_list", response_model=HistoryListResponse)
async def get_history_list():
    sessions = []
    # 获取所有会话 ID
    session_keys = redis_client.keys('session:*')
    for key in session_keys:
        session_data = redis_client.hgetall(key)
        session = {
            "sessionId": session_data[b'sessionId'].decode('utf-8'),
            "name": session_data[b'name'].decode('utf-8'),
            "content": ""
        }
        sessions.append(session)
    return {
        "data": {
            "templates": sessions
        }
    }


# 2. 获取历史详细对话接口
@router.post("/chat/history_detail", response_model=HistoryDetailResponse)
async def get_history_detail(request: Request):
    data = await request.json()
    session_id = data.get('sessionId')
    message_key = f"messages:{session_id}"
    messages = []
    if redis_client.exists(message_key):
        message_list = redis_client.lrange(message_key, 0, -1)
        for msg in message_list:
            messages.append(json.loads(msg.decode('utf-8')))
    return {
        "data": {
            "messages": messages
        }
    }


# 3. 上传图片接口
@router.post("/chat/upload_image", response_model=UploadImageResponse)
async def upload_image(request: UploadImageRequest):
    try:
        image_list = request.image_list
        # 这里可以添加图片保存逻辑，例如保存到本地或云存储
        for image_info in image_list:
            image_type = image_info.get('type')
            image = image_info.get('image')
            # 模拟保存成功
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



# 5. 发送消息接口
@router.post("/chat/send")
async def send_message(request: SendMessageRequest):
    session_id = request.sessionId
    message = request.message

    session_key = f"session:{session_id}"
    if not redis_client.exists(session_key):
        raise HTTPException(status_code=400, detail="Invalid session ID")

    user_message = {
        "role": "user",
        "content": message,
        "attachments": []
    }
    message_key = f"messages:{session_id}"
    redis_client.rpush(message_key, json.dumps(user_message))

    message_id = str(uuid.uuid4())

    async def generate_response():
        response_chunks = ["地震", "发生后", "，我建议"]
        for chunk in response_chunks:
            data = {
                "message_id": message_id,
                "data": {
                    "content": chunk,
                    "done": False
                }
            }
            yield f"data: {str(data).replace(' ', '')}\n\n"
            await asyncio.sleep(1)

        assistant_message = {
            "role": "assistant",
            "content": "".join(response_chunks),
            "attachments": []
        }
        redis_client.rpush(message_key, json.dumps(assistant_message))

        end_data = {
            "message_id": message_id,
            "data": {
                "done": True
            }
        }
        yield f"data: {str(end_data).replace(' ', '')}\n\n"

    return StreamingResponse(generate_response(), media_type='text/event-stream')


# 6. 预设提示接口
@router.post("/prompts/templates", response_model=PromptsResponse)
async def get_prompts():
    return {
        "data": {
            "templates": [
                {
                    "id": "disaster_summary",
                    "name": "分析整体灾情",
                    "content": "根据灾后影像详细分析一下整体灾情"
                }
            ]
        }
    }