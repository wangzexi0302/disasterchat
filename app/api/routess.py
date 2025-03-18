from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from app.api.database.db_setup import get_db
from app.api.database.models import DBSession, ChatMessage, Image, MessageImage, PromptTemplate
from pydantic import BaseModel
import redis
import json
import uuid
import asyncio
from fastapi.responses import StreamingResponse

# 初始化 Redis 客户端，用于缓存数据和实现异步操作
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 创建一个 FastAPI 的路由实例，设置路由前缀和标签
router = APIRouter(prefix="/api", tags=["chat"])


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


# 发送消息接口的请求模型
class SendMessageRequest(BaseModel):
    # 会话 ID
    sessionId: str
    # 消息内容
    message: str
    # 关联的图片 ID 列表，默认为空列表
    image_ids: list[str] = []


# 获取历史详细对话接口的响应模型
class HistoryDetailResponse(BaseModel):
    # 包含历史详细对话的数据
    data: dict


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


# ======================
# 2. 核心接口实现
# 这里定义了各个接口的具体实现逻辑
# ======================

# === 开启新会话接口 ===
@router.post("/chat/new_session", response_model=NewSessionResponse)
async def new_session(
    background_tasks: BackgroundTasks,  # 用于添加后台任务，实现异步操作
    db: Session = Depends(get_db)  # 依赖注入 SQLAlchemy 的数据库会话
):
    # 生成一个唯一的会话 ID
    session_id = str(uuid.uuid4())

    # 使用 Redis 的原子计数器生成会话名称，确保名称的唯一性
    session_number = redis_client.incr("session_counter")
    session_name = f"会话 {session_number}"

    # 将会话信息写入 Redis
    session_key = f"session:{session_id}"
    redis_client.hset(session_key, mapping={
        "sessionId": session_id,
        "name": session_name
    })

    # 定义一个异步写入 MySQL 的函数
    def save_to_mysql():
        try:
            # 创建一个新的会话记录
            db_session = DBSession(
                id=session_id,
                name=session_name
            )
            # 将新会话记录添加到数据库会话中
            db.add(db_session)
            # 提交数据库事务
            db.commit()
        except Exception as e:
            # 若出现异常，回滚数据库事务
            db.rollback()
            # 抛出 HTTP 异常，返回 500 错误
            raise HTTPException(500, detail="数据库写入失败")
        finally:
            # 关闭数据库会话
            db.close()

    # 将异步写入 MySQL 的任务添加到后台任务中
    background_tasks.add_task(save_to_mysql)

    # 返回包含新会话 ID 的响应
    return {"session_id": session_id}


# === 获取历史对话列表 ===
@router.post("/chat/history_list", response_model=HistoryListResponse)
async def get_history_list(db: Session = Depends(get_db)):
    # 尝试从 Redis 缓存中获取历史对话列表
    cached_sessions = redis_client.get("history_sessions")
    if cached_sessions:
        # 若缓存存在，直接返回缓存数据
        return {"data": {"templates": json.loads(cached_sessions)}}

    # 若缓存不存在，从 MySQL 数据库中查询所有会话记录
    db_sessions = db.query(DBSession).all()
    # 处理查询结果，将其转换为需要的格式
    sessions = [
        {"sessionId": s.id, "name": s.name, "content": ""}
        for s in db_sessions
    ]

    # 将查询结果更新到 Redis 缓存中，并设置缓存过期时间为 1 小时
    redis_client.set("history_sessions", json.dumps(sessions), ex=3600)
    # 返回包含历史对话列表的响应
    return {"data": {"templates": sessions}}


# === 发送消息接口 ===
@router.post("/chat/send")
async def send_message(
    request: SendMessageRequest,  # 接收发送消息的请求数据
    db: Session = Depends(get_db)  # 依赖注入 SQLAlchemy 的数据库会话
):
    # 验证会话是否存在于 Redis 中
    if not redis_client.exists(f"session:{request.sessionId}"):
        # 若会话不存在，抛出 HTTP 异常，返回 400 错误
        raise HTTPException(400, "会话不存在")

    # 构建用户消息的字典
    message_key = f"messages:{request.sessionId}"
    user_msg = {
        "role": "user",
        "content": request.message,
        "image_ids": request.image_ids
    }
    # 将用户消息添加到 Redis 中
    redis_client.rpush(message_key, json.dumps(user_msg))

    # 定义一个异步保存消息到 MySQL 的协程函数
    async def save_messages():
        # 创建一个新的消息记录
        db_msg = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=request.sessionId,
            role="user",
            content=request.message,
            attachments=json.dumps(request.image_ids)
        )
        # 将新消息记录添加到数据库会话中
        db.add(db_msg)

        # 处理消息关联的图片
        for img_id in request.image_ids:
            # 查询图片记录
            db_image = Image.query.get(img_id)
            if not db_image:
                continue
            # 创建消息和图片的关联记录
            db_rel = MessageImage(
                message_id=db_msg.id,
                image_id=img_id
            )
            # 将关联记录添加到数据库会话中
            db.add(db_rel)

        # 提交数据库事务
        db.commit()

    # 异步执行保存消息到 MySQL 的任务
    asyncio.create_task(save_messages())

    # 定义一个流式响应的协程函数
    async def stream_response():
        # 发送流式响应的第一部分，表示正在思考
        yield 'data: {"content": "思考中...", "done": false}\n\n'
        # 模拟处理时间
        await asyncio.sleep(2)
        # 发送流式响应的第二部分，表示处理完成
        yield 'data: {"content": "完成！", "done": true}\n\n'

    # 返回流式响应
    return StreamingResponse(stream_response(), media_type="text/event-stream")


# === 获取历史详细对话 ===
@router.post("/chat/history_detail", response_model=HistoryDetailResponse)
async def get_history_detail(
    request: SendMessageRequest,  # 接收获取历史详细对话的请求数据
    db: Session = Depends(get_db)  # 依赖注入 SQLAlchemy 的数据库会话
):
    # 尝试从 Redis 缓存中获取历史详细对话
    cached_msgs = redis_client.get(f"history:{request.sessionId}")
    if cached_msgs:
        # 若缓存存在，直接返回缓存数据
        return {"data": {"messages": json.loads(cached_msgs)}}

    # 若缓存不存在，从 MySQL 数据库中查询指定会话的所有消息记录
    db_msgs = db.query(ChatMessage).filter(
        ChatMessage.session_id == request.sessionId
    ).all()

    # 处理查询结果，将其转换为需要的格式
    messages = []
    for msg in db_msgs:
        messages.append({
            "role": msg.role,
            "content": msg.content,
            "image_ids": json.loads(msg.attachments)
        })

    # 将查询结果更新到 Redis 缓存中，并设置缓存过期时间为 1 小时
    redis_client.set(f"history:{request.sessionId}", json.dumps(messages), ex=3600)
    # 返回包含历史详细对话的响应
    return {"data": {"messages": messages}}


# === 上传图片接口 ===
@router.post("/chat/upload_image", response_model=UploadImageResponse)
async def upload_image(
    files: list[UploadFile] = File(...),  # 接收上传的图片文件列表
    db: Session = Depends(get_db)  # 依赖注入 SQLAlchemy 的数据库会话
):
    # 检查是否提供了图片文件
    if not files:
        # 若未提供，抛出 HTTP 异常，返回 400 错误
        raise HTTPException(400, "未提供图片")

    # 用于存储上传成功的图片 ID
    image_ids = []
    for file in files:
        # 生成一个唯一的文件名，用于保存图片
        save_path = f"uploads/{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        # 以二进制写入模式打开文件，并将上传的图片内容写入
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # 创建一个新的图片记录
        db_image = Image(
            id=str(uuid.uuid4()),
            file_path=save_path,
            file_type=file.content_type
        )
        # 将新图片记录添加到数据库会话中
        db.add(db_image)
        # 将图片 ID 添加到上传成功的图片 ID 列表中
        image_ids.append(db_image.id)

    # 提交数据库事务
    db.commit()
    # 返回包含上传状态、消息和图片 ID 列表的响应
    return UploadImageResponse(
        status="success",
        image_ids=image_ids,
        message=f"成功上传 {len(files)} 张图片"
    )


# === 预设提示接口 ===
@router.post("/prompts/templates", response_model=PromptsResponse)
async def get_prompts(db: Session = Depends(get_db)):
    # 尝试从 Redis 缓存中获取预设提示模板
    cached_prompts = redis_client.get("prompts")
    if cached_prompts:
        # 若缓存存在，直接返回缓存数据
        return {"data": {"templates": json.loads(cached_prompts)}}

    # 若缓存不存在，从 MySQL 数据库中查询所有预设提示模板记录
    db_prompts = db.query(PromptTemplate).all()
    # 处理查询结果，将其转换为需要的格式
    prompts = [
        {"id": p.id, "name": p.name, "content": p.content}
        for p in db_prompts
    ]

    # 将查询结果更新到 Redis 缓存中，并设置缓存过期时间为 1 小时
    redis_client.set("prompts", json.dumps(prompts), ex=3600)
    # 返回包含预设提示模板的响应
    return {"data": {"templates": prompts}}
