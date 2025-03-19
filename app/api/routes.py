from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from app.api.database.db_setup import get_db
from app.api.database.models import DBSession, ChatMessage, Image, MessageImage, PromptTemplate
from pydantic import BaseModel
import redis
import json
import uuid
import asyncio
from fastapi.responses import StreamingResponse # 流式响应
from app.api.models import ChatRequest, ChatResponse, VLChatMessage,ImageContent,TextContent,SendMessageRequest
from app.agents.agent_service import AgentService
from app.agents.multimodal_agent import MultiModalAgent

# 初始化 Redis 客户端，用于缓存数据和实现异步操作
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 创建一个 FastAPI 的路由实例，设置路由前缀和标签
router = APIRouter(prefix="/api", tags=["chat"])

# 初始化 LLM 服务
agent_service = AgentService()
multimodal_agent = MultiModalAgent()

# ======================
# 1. 公共模型定义
# 这里定义了各个接口所使用的请求和响应数据模型，使用 Pydantic 进行数据验证
# ======================

# 开启新会话接口的响应模型
class NewSessionResponse(BaseModel):
    # 操作状态，默认为成功
    status: str = "success"
    # 新创建的会话 ID
    session_id: str

# 获取历史对话列表接口的响应模型
class HistoryListResponse(BaseModel):
    # 包含历史对话列表的数据
    data: dict

# 获取历史详细对话接口的响应模型
class HistoryDetailResponse(BaseModel):
    data: dict

    class Config:
        arbitrary_types_allowed = True
    @classmethod
    def from_orm(cls, obj):
        messages = []
        for db_message in obj["messages"]:
            content = []
            if db_message.attachments:
                image_ids = json.loads(db_message.attachments)
                for image_id in image_ids:
                    image = db.query(Image).filter(Image.id == image_id).first()
                    if image:
                        content.append(ImageContent(image_id=image_id))
            if db_message.content:
                content.append(TextContent(text=db_message.content))
            message = VLChatMessage(
                role=db_message.role,
                content=content if len(content) > 1 else content[0] if content else ""
            )
            messages.append(message)
        return cls(data={"messages": messages})


# 上传图片接口的响应模型
class UploadImageResponse(BaseModel):
    # 操作状态
    status: str
    # 操作消息，可选
    message: str = None
    # 上传成功的图片 ID 列表
    image_ids: list[str] = []


# 预设提示接口的响应模型
class PromptsResponse(BaseModel):
    # 包含预设提示模板的数据
    data: dict

# 辅助函数
def save_session_to_mysql(session_id, session_name, db):
    try:
        db_session = DBSession(
            id=session_id,
            name=session_name
        )
        db.add(db_session)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail="数据库写入失败")
    finally:
        db.close()


def get_cached_or_db_data(cache_key, db_query, db, cache_time=3600):
    cached_data = redis_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    db_data = db_query.all()
    data = [
        {"sessionId": s.id, "name": s.name, "content": ""} if isinstance(s, DBSession) else
        {"id": p.id, "name": p.name, "content": p.content} if isinstance(p, PromptTemplate) else
        {} for s in db_data
    ]
    redis_client.set(cache_key, json.dumps(data), ex=cache_time)
    return data


# 核心接口实现
@router.post("/chat/new_session", response_model=NewSessionResponse)
async def new_session(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())
    session_number = redis_client.incr("session_counter")
    session_name = f"会话 {session_number}"
    session_key = f"session:{session_id}"
    redis_client.hset(session_key, mapping={
        "sessionId": session_id,
        "name": session_name
    })
    background_tasks.add_task(save_session_to_mysql, session_id, session_name, db)
    return {"session_id": session_id}


@router.post("/chat/history_list", response_model=HistoryListResponse)
async def get_history_list(db: Session = Depends(get_db)):
    sessions = get_cached_or_db_data("history_sessions", db.query(DBSession), db)
    return {"data": {"templates": sessions}}


@router.post("/chat/send")
async def send_message(request: SendMessageRequest, db: Session = Depends(get_db)):
    session_id = request.sessionId
    message = request.message
    session_key = f"session:{session_id}"
    if not redis_client.exists(session_key):
        raise HTTPException(status_code=400, detail="Invalid session ID")
    db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    text_content = ""
    image_ids = []
    if isinstance(message.content, str):
        text_content = message.content
    else:
        for item in message.content:
            if item.type == "text":
                text_content += item.text
            elif item.type == "image":
                image_ids.append(item.image_id)

    user_message = {
        "role": message.role,
        "content": text_content,
        "attachments": image_ids
    }
    message_key = f"messages:{session_id}"
    redis_client.rpush(message_key, json.dumps(user_message))

    user_db_message = ChatMessage(
        session_id=session_id,
        role=message.role,
        content=text_content,
        attachments=json.dumps(image_ids)
    )
    db.add(user_db_message)
    db.commit()
    db.refresh(user_db_message)

    for image_id in image_ids:
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        message_image = MessageImage(
            message_id=user_db_message.id,
            image_id=image.id
        )
        db.add(message_image)
    db.commit()

    message_id = str(uuid.uuid4())

    # 判断是否为多模态消息
    is_multimodal = isinstance(message.content, list) and any(
        isinstance(item, ImageContent) for item in message.content
    )

    if is_multimodal:
        # 多模态消息调用
        processed_messages = [message.dict()]
        try:
            async def generate_multimodal_response():
                stream_response = multimodal_agent.run_stream(processed_messages)
                chunk_count = 0
                for chunk in stream_response:
                    chunk_count += 1
                    content = chunk.get("message", {}).get("content", "")
                    data = {
                        "message_id": message_id,
                        "data": {
                            "content": content,
                            "done": False
                        }
                    }
                    yield f"data: {str(data).replace(' ', '')}\n\n"
                end_data = {
                    "message_id": message_id,
                    "data": {
                        "done": True
                    }
                }
                yield f"data: {str(end_data).replace(' ', '')}\n\n"
            return StreamingResponse(generate_multimodal_response(), media_type='text/event-stream')
        except Exception as e:
            raise HTTPException(500, f"多模态推理失败: {str(e)}")
    else:
        # 文本消息调用
        try:
            async def generate_text_response():
                stream_response = agent_service.run_stream([{"role": message.role, "content": text_content}], "gpt-4-vision")
                for chunk in stream_response:
                    content = chunk.get("message", {}).get("content", "")
                    data = {
                        "message_id": message_id,
                        "data": {
                            "content": content,
                            "done": False
                        }
                    }
                    yield f"data: {str(data).replace(' ', '')}\n\n"
                end_data = {
                    "message_id": message_id,
                    "data": {
                        "done": True
                    }
                }
                yield f"data: {str(end_data).replace(' ', '')}\n\n"
            return StreamingResponse(generate_text_response(), media_type='text/event-stream')
        except Exception as e:
            raise HTTPException(500, f"文本推理失败: {str(e)}")


@router.post("/chat/history_detail", response_model=HistoryDetailResponse)
async def get_history_detail(request: SendMessageRequest, db: Session = Depends(get_db)):
    session_id = request.sessionId
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing sessionId in request")
    cached_messages = redis_client.get(f"history_messages:{session_id}")
    if cached_messages:
        messages = json.loads(cached_messages)
    else:
        db_messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
        messages = []
        for db_message in db_messages:
            image_relations = db.query(MessageImage).filter(MessageImage.message_id == db_message.id).all()
            images = []
            for relation in image_relations:
                image = db.query(Image).filter(Image.id == relation.image_id).first()
                if image:
                    images.append({
                        "id": image.id,
                        "file_path": image.file_path,
                        "file_type": image.file_type
                    })
            message = {
                "role": db_message.role,
                "content": db_message.content,
                "attachments": json.loads(db_message.attachments) if db_message.attachments else [],
                "images": images
            }
            messages.append(message)
        redis_client.set(f"history_messages:{session_id}", json.dumps(messages), ex=3600)
    return HistoryDetailResponse.from_orm({"messages": db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()})


@router.post("/chat/upload_image", response_model=UploadImageResponse)
async def upload_image(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    if not files:
        raise HTTPException(400, "未提供图片")
    image_ids = []
    for file in files:
        save_path = f"uploads/{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        with open(save_path, "wb") as f:
            f.write(await file.read())
        db_image = Image(
            id=str(uuid.uuid4()),
            file_path=save_path,
            file_type=file.content_type
        )
        db.add(db_image)
        image_ids.append(db_image.id)
    db.commit()
    return UploadImageResponse(
        status="success",
        image_ids=image_ids,
        message=f"成功上传 {len(files)} 张图片"
    )


@router.post("/prompts/templates", response_model=PromptsResponse)
async def get_prompts(db: Session = Depends(get_db)):
    prompts = get_cached_or_db_data("prompts", db.query(PromptTemplate), db)
    return {"data": {"templates": prompts}}