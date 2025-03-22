from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import logging
from app.api.models import ChatRequest, ChatResponse, VLChatMessage
from app.agents.agent_service import AgentService
from app.agents.multimodal_agent import MultiModalAgent
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["llm"])

# 创建Agent服务实例
agent_service = AgentService()
multimodal_agent = MultiModalAgent()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    与LLM Agent交互的端点
    """
    try:
        model = request.model or settings.default_model
        logger.info(f"接收到聊天请求，使用模型：{model}")
        
        # 记录用户最新消息
        if request.messages:
            latest_msg = request.messages[-1].content
            logger.info(f"用户查询: {latest_msg[:100]}{'...' if len(latest_msg) > 100 else ''}")
        
        response = agent_service.run(
            messages=[msg.dict() for msg in request.messages], 
            model=model
        )
        
        logger.info("成功生成回复")
        return response
    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    与LLM Agent交互的流式端点
    """
    try:
        model = request.model or settings.default_model
        logger.info(f"接收到流式聊天请求，使用模型：{model}")
        
        # 记录用户最新消息
        if request.messages:
            latest_msg = request.messages[-1].content
            logger.info(f"用户流式查询: {latest_msg[:100]}{'...' if len(latest_msg) > 100 else ''}")
        
        async def generate():
            try:
                for chunk in agent_service.run_stream(
                    messages=[msg.dict() for msg in request.messages],
                    model=model
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # 标记流式输出结束
                logger.info("流式回复生成完成")
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error(f"流式生成过程中出错: {str(e)}", exc_info=True)
                # 在流中标记错误
                error_msg = {"error": str(e)}
                yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"流式聊天请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """健康检查端点"""
    logger.debug("接收到健康检查请求")
    return {"status": "ok"}


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
async def send_message(request: SendMessageRequest, db: Session = Depends(get_db)):
    session_id = request.sessionId
    message = request.message

    session_key = f"session:{session_id}"
    if not redis_client.exists(session_key):
        raise HTTPException(status_code=400, detail="Invalid session ID")

    # 检查会话是否存在于 MySQL 中
    db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    # 处理消息内容
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

    # 构造用户消息
    user_message = {
        "role": message.role,
        "content": text_content,
        "attachments": image_ids
    }
    message_key = f"messages:{session_id}"
    redis_client.rpush(message_key, json.dumps(user_message))

    # 将用户消息存入 MySQL
    user_db_message = ChatMessage(
        session_id=session_id,
        role=message.role,
        content=text_content,
        attachments=json.dumps(image_ids)
    )
    db.add(user_db_message)
    db.commit()
    db.refresh(user_db_message)

    # 绑定消息与图片
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

        assistant_content = "".join(response_chunks)
        assistant_message = {
            "role": "assistant",
            "content": assistant_content,
            "attachments": []
        }
        redis_client.rpush(message_key, json.dumps(assistant_message))

        # 将助手消息存入 MySQL
        assistant_db_message = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=assistant_content,
            attachments=json.dumps([])
        )
        db.add(assistant_db_message)
        db.commit()
        db.refresh(assistant_db_message)

        end_data = {
            "message_id": message_id,
            "data": {
                "done": True
            }
        }
        yield f"data: {str(end_data).replace(' ', '')}\n\n"

    return StreamingResponse(generate_response(), media_type='text/event-stream')


# === 获取历史详细对话 ===
@router.post("/chat/history_detail", response_model=HistoryDetailResponse)
async def get_history_detail(request: SendMessageRequest, db: Session = Depends(get_db)):
    session_id = request.sessionId

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing sessionId in request")

    # 先尝试从 Redis 中获取缓存数据
    cached_messages = redis_client.get(f"history_messages:{session_id}")
    if cached_messages:
        messages = json.loads(cached_messages)
    else:
        # 从 MySQL 中获取该会话的所有消息
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

        # 将数据存入 Redis 缓存
        redis_client.set(f"history_messages:{session_id}", json.dumps(messages), ex=3600)  # 缓存 1 小时

    return HistoryDetailResponse.from_orm({"messages": db_messages})

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


@router.post("/chat/send")
async def send_message(
    request: SendMessageRequest,request_obj: Request,db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    优化后的消息发送接口，支持文本/多模态消息流式响应及SSE实时推送
    """
    try:
        # ----------------------
        # 1. 会话验证（Redis优先）
        # ----------------------
        session_id = request.sessionId
        session_key = f"session:{session_id}"

        # 快速验证会话（Redis）
        if not redis_client.exists(session_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID"
            )
        # 验证数据库会话
        db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if not db_session:
            raise HTTPException(400, "Invalid session ID in database")

        # ----------------------
        # 2. 解析消息内容
        # ----------------------
        text_content = ""
        image_ids = []
        is_multimodal = False

        # 处理消息内容
        if isinstance(request.message.content, str):
            text_content = request.message.content.strip()
        elif isinstance(request.message.content, list):
            for item in request.message.content:
                if item.type == "text":
                    text_content += item.text.strip() + " "
                elif item.type == "image":
                    if not item.image_id:
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Image content missing image_id"
                        )
                    image_ids.append(item.image_id)
                    is_multimodal = True
            text_content = text_content.strip()  # 去除多余空格
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid content type. Expected str or list"
            )

        # ----------------------
        # 3. 数据库事务处理
        # ----------------------
        with db.begin_nested():  # 开启事务嵌套
            # 主消息记录
            chat_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=request.message.role,
                content=text_content,
                attachments=json.dumps(image_ids) if image_ids else None
            )
            db.add(chat_message)

            # 验证并关联图片
            image_paths = []
            for img_id in image_ids:
                db_image = db.query(Image).filter(Image.id == img_id).first()
                if not db_image:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Image not found: {img_id}"
                    )
                image_paths.append(db_image.file_path)  # 收集图片路径

                # 消息-图片关联
                db.add(MessageImage(
                    message_id=chat_message.id,
                    image_id=img_id
                ))

        # ----------------------
        # 4. 构造LLM消息格式
        # ----------------------
        llm_messages = []
        if is_multimodal:
            # 多模态消息格式（文本+图片路径）
            llm_messages = [{
                "role": request.message.role,
                "content": [
                    {"type": "text", "text": text_content},
                    *[{"type": "image", "image_data": path} for path in image_paths]
                ]
            }]
        else:
            # 纯文本消息
            llm_messages = [{"role": request.message.role, "content": text_content}]

        # ----------------------
        # 5. 流式响应生成及SSE推送
        # ----------------------
        async def sse_stream_generator() -> Generator[str, None, None]:
            try:
                # 选择代理
                if is_multimodal:
                    stream_response = multimodal_agent.run_stream(llm_messages, model="llava:latest")
                else:
                    stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")

                message_id = str(uuid.uuid4())

                # 创建AI消息记录
                assistant_message = ChatMessage(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content="",  # 后续流式填充内容
                    attachments=None
                )
                db.add(assistant_message)

                for chunk in stream_response:
                    logger.info(f"原始 chunk: {chunk}")  # 关键调试日志
                    content = (
                        chunk.get("message", {}).get("content", "")  # Ollama 格式
                    )
                    if not content:
                        content = chunk.get("text", "") or chunk.get("output", "") or ""
                    if content.strip():
                        assistant_message.content += content
                        sse_chunk = {
                            "message_id": message_id,
                            "data": {
                                "content": content,
                                "done": chunk.get("done", False)
                            }
                        }
                        # 发送SSE消息
                        yield f"data: {json.dumps(sse_chunk)}\n\n"
                        # 推送到Redis供其他客户端接收
                        redis_client.rpush(
                            f"messages:{session_id}",
                            json.dumps({
                                "id": message_id,
                                "role": "assistant",
                                "content": content,
                                "attachments": [],
                                "created_at": datetime.utcnow().isoformat()
                            })
                        )
                logger.info("流式回复生成完成")
                db.commit()
                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

            except Exception as e:
                error_chunk = {
                    "error": str(e),
                    "detail": "Inference failed"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        # ----------------------
        # 6. 保存用户消息到Redis
        # ----------------------
        if redis_client:
            redis_client.rpush(
                f"messages:{session_id}",
                json.dumps({
                    "id": chat_message.id,
                    "role": request.message.role,
                    "content": text_content,
                    "attachments": image_paths,
                    "created_at": datetime.utcnow().isoformat()
                })
            )

        return StreamingResponse(
            sse_stream_generator(),
            media_type="text/event-stream"
        )

    except HTTPException as e:
        db.rollback()  # 回滚事务
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )
    
    
        with db.begin_nested():  # 开启事务嵌套
            # 主消息记录
            chat_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=request.message.role,
                content=text_content,
                attachments=json.dumps(image_ids) if image_ids else None
            )
            db.add(chat_message)

            # 验证并关联图片
            image_paths = []
            for img_id in image_ids:
                db_image = db.query(Image).filter(Image.id == img_id).first()
                if not db_image:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Image not found: {img_id}"
                    )
                image_paths.append(db_image.file_path)  # 收集图片路径

                # 消息-图片关联
                db.add(MessageImage(
                    message_id=chat_message.id,
                    image_id=img_id
                ))