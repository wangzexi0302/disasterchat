import random
from time import sleep
import time
from fastapi import APIRouter,Request,Depends,Form, BackgroundTasks, HTTPException,UploadFile,File,status
from sqlalchemy.orm import Session
from app.api.database.db_setup import get_db
from app.api.database.models import DBSession, ChatMessage, Image, MessageImage, PromptTemplate
from pydantic import BaseModel
from datetime import datetime  # 添加此行
from typing import Generator, List, Dict, Any
import redis
import json
from sqlalchemy import orm
import uuid
import asyncio
from fastapi.responses import StreamingResponse # 流式响应
from app.api.models import ChatRequest, ChatResponse, VLChatMessage,ImageContent,TextContent,SendMessageRequest
from app.agents.agent_service import AgentService
from app.agents.multimodal_agent import MultiModalAgent
from app.api.database.db_setup import engine
from app.api.database.models import Base
import os
import logging

logger = logging.getLogger(__name__)  # 使用模块级 logger



Base.metadata.create_all(bind=engine)

# 删除所有表（生产环境禁用！）
#Base.metadata.drop_all(engine)

# 重新创建表
#Base.metadata.create_all(engine)

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
class ImageAttachment(BaseModel):
    url: str       # 图片URL（由file_path生成）
    type: str      # 图片类型（如image/jpeg）

class ChatMessageResponse(BaseModel):
    id: str                # 消息ID
    role: str              # user/assistant
    content: str           # 文本内容
    attachments: List[ImageAttachment]  # 图片附件列表
    created_at: str        # ISO格式时间

class HistoryDetailResponse(BaseModel):
    data: Dict[str, List[ChatMessageResponse]]

class HistoryDetailRequest(BaseModel):
    sessionId: str  # 仅需会话ID 

def generate_image_id(image_url):
    # 获取当前时间戳
    timestamp = str(int(time.time()))
    # 生成一个随机整数
    random_num = str(random.randint(1000, 9999))
    # 拼接时间戳和随机数作为时间片随机值
    random_value = timestamp + random_num
    # 拼接图片 URL 和时间片随机值
    image_id = image_url + random_value
    return image_id

# 上传图片接口的响应模型
class UploadImageResponse(BaseModel):
    # 操作状态
    status: str
    # 操作消息，可选
    message: str = None
    # 上传成功的图片 ID 列表
    image_ids: list[str] = []

class TemplateItem(BaseModel):
    id: str
    name: str
    content: str

class PromptTemplatesResponse(BaseModel):
    data: Dict[str, List[TemplateItem]]

# 预设提示模板（可从数据库或配置文件加载）
PREDEFINED_TEMPLATES = [
    TemplateItem(
        id="disaster_summary",
        name="分析整体灾情",
        content="请告诉我灾后影像的大致受灾情况"
    ),
    TemplateItem(
        id="relief_suggestion",
        name="救援建议",
        content="请提供具体的灾后救援建议和资源分配方案"
    )
]





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
#开启新会话
@router.post("/chat/new_session", response_model=NewSessionResponse)
async def new_session(db: Session = Depends(get_db)):
    try:
        session_id = str(uuid.uuid4())
        session_number = redis_client.incr("session_counter")
        session_name = f"会话 {session_number}"
        session_key = f"session:{session_id}"

        # 同步存入 Redis
        redis_client.hset(session_key, mapping={
            "sessionId": session_id,
            "name": session_name
        })

        # 存入 MySQL
        db_session = DBSession(
            id=session_id,
            name=session_name
        )
        db.add(db_session)
        db.commit()

        return {"session_id": session_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=f"数据库写入失败: {str(e)}")
    finally:
        db.close()

# 修改会话名称接口
@router.post("/chat/update_session_name", response_model=NewSessionResponse)
async def update_session_name(
    session_id: str = Form(...),
    new_name: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        session_key = f"session:{session_id}"
        # 验证会话是否存在于 Redis
        if not redis_client.exists(session_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID"
            )
        # 验证数据库会话
        db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if not db_session:
            logging.error(f"会话id不存在: {session_id}")
            raise HTTPException(400, "Invalid session ID in database")

        # 更新数据库中的会话名称
        db_session.name = new_name
        db.commit()

        # 更新 Redis 中的会话名称
        redis_client.hset(session_key, "name", new_name)

        return {"session_id": session_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=f"更新会话名称失败: {str(e)}")
    finally:
        db.close()

# 删除会话接口
@router.post("/chat/delete_session", response_model=NewSessionResponse)
async def delete_session(
    session_id: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        session_key = f"session:{session_id}"
        # 验证会话是否存在于 Redis
        if not redis_client.exists(session_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID"
            )
        # 验证数据库会话
        db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if not db_session:
            logging.error(f"会话id不存在: {session_id}")
            raise HTTPException(400, "Invalid session ID in database")

        # 删除数据库中的会话记录
        db.delete(db_session)
        db.commit()

        # 删除 Redis 中的会话信息
        redis_client.delete(session_key)

        return {"session_id": session_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=f"删除会话失败: {str(e)}")
    finally:
        db.close()

#查询历史会话接口
@router.post("/chat/history_list", response_model=HistoryListResponse)
async def get_history_list(db: Session = Depends(get_db)):
    sessions = get_cached_or_db_data("history_sessions", db.query(DBSession), db)
    return {"data": {"templates": sessions}}

#历史消息详情接口
@router.post("/chat/history_detail", response_model=HistoryDetailResponse)
async def get_history_detail(
    request: HistoryDetailRequest,  # 专用请求模型
    db: orm.Session = Depends(get_db)
):
    session_id = request.sessionId
    if not session_id:
        raise HTTPException(400, "会话ID不能为空")
    

    # 1. 缓存优先（Redis）
    cache_key = f"history:{session_id}"
    cached_data = redis_client.get(cache_key)
    if cached_data:
        return HistoryDetailResponse(**json.loads(cached_data))

    # 2. 数据库查询（JOIN优化）
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc()).all()

    response_messages = []
    for msg in messages:
        # 使用JOIN一次性获取消息关联的图片（避免N+1查询）
        images = db.query(Image).join(MessageImage).filter(
            MessageImage.message_id == msg.id
        ).all()

        response_messages.append(ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            attachments=[
                ImageAttachment(
                    url=f"http://your-domain.com/{img.file_path}",  # 替换为实际URL前缀
                    type=img.file_type
                ) for img in images
            ],
            created_at=msg.created_at.isoformat()
        ))

    # 3. 缓存结果（有效期1小时）
    redis_client.set(cache_key, json.dumps(HistoryDetailResponse(
        data={"messages": response_messages}
    ).dict()), ex=3600)

    return HistoryDetailResponse(data={"messages": response_messages})
#图片上传接口
@router.post("/chat/upload_image", response_model=UploadImageResponse)
async def upload_image(type: str = Form(...), files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    if not files:
        raise HTTPException(400, "未提供图片")
    image_ids = []
    upload_dir = "uploads"
    # 检查上传目录是否存在，不存在则创建
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    for file in files:
        try:
            # 生成保存路径
            file_extension = file.filename.split('.')[-1]
            save_path = os.path.join(upload_dir, f"{uuid.uuid4()}.{file_extension}")
            # 写入文件
            with open(save_path, "wb") as f:
                f.write(await file.read())
            # 创建数据库记录
            db_image = Image(
                id=str(uuid.uuid4()),
                file_path=save_path,
                file_type=file.content_type,
                type=type  # 新增字段存储 type 参数
            )
            db.add(db_image)
            image_ids.append(db_image.id)
        except Exception as e:
            # 发生错误时进行回滚操作
            db.rollback()
            raise HTTPException(500, f"上传图片时发生错误: {str(e)}")

    try:
        # 提交数据库事务
        db.commit()
    except Exception as e:
        # 数据库提交失败时进行回滚操作
        db.rollback()
        raise HTTPException(500, f"保存图片信息到数据库时发生错误: {str(e)}")

    return UploadImageResponse(
        status="success",
        image_ids=image_ids,
        message=f"成功上传 {len(files)} 张图片"
    )

#预设提示接口
@router.post("/prompts/templates", response_model=PromptTemplatesResponse, status_code=status.HTTP_200_OK)
async def get_prompt_templates():
    """
    获取预设提示模板列表
    - **无参数**
    - **响应**：包含模板ID、名称和内容的列表
    """
    return {
        "data": {
            "templates": PREDEFINED_TEMPLATES
        }
    }
#发送消息接口
@router.post("/chat/send")
async def send_message(
    request: SendMessageRequest,
    request_obj: Request,
    db: Session = Depends(get_db)
) -> StreamingResponse:
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
            logging.error(f"会话id不存在: {session_id}")
            raise HTTPException(400, "Invalid session ID in database")
        else:
            db.close()

        

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
        try:
            if db.in_transaction():
                logging.error("事务已经开启，无法再次开启！")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="事务已经开启，无法再次开启！"
                )
            with db.begin():
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
            #db.commit()  # 手动提交

        except HTTPException as e:
            db.rollback()  # 回滚事务
            logging.error(f"回滚HTTPException: {e}")
            raise e
        except Exception as e:
            db.rollback()
            logging.error(f"回滚Unexpected error: {e}. Request body: {request.dict()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Server error: {str(e)}"
            )
        finally:
            # 在事务结束后记录数据库会话状态
            logging.info(f"Database 最终session: {db.is_active}")

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

        # 检查是否为“变化检测”
        if text_content == "变化检测":
            async def sse_stream_generator() -> Generator[str, None, None]:
                # 调用大模型
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

                # 构造图片URL
                image_url = str(request_obj.url_for('static', path="test_image_2.png"))

                ## 给一个image_list可以拓展
                image_list = [image_url]
                for image in image_list:
                    image_id = generate_image_id(image_url)
                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image,'image_id':image_id,'type':"post"}})}\n\n"
                
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
                                "done": chunk.get("done", False),
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
                                "attachments": [image_url],
                                "created_at": datetime.utcnow().isoformat()
                            })
                        )
                
                logger.info("流式回复生成完成")
                db.commit()
                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"
            return StreamingResponse(
                sse_stream_generator(),
                media_type="text/event-stream"
            )
        elif text_content == "请告诉我灾后影像的大致受灾情况":
            async def sse_stream_generator() -> Generator[str, None, None]:
                message_id = str(uuid.uuid4())
                assistant_message = ChatMessage(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content="",  # 后续流式填充内容
                    attachments=None
                )
                db.add(assistant_message)
                response_text = """这张影像显示了一个受灾区域，飓风后的卫星图像。从图像来看，主要的受灾情况包括:
大面积积水:图像显示大片的浑浊水域，覆盖了树林和部分居民区。这表明该区域可能经历了严重的洪水，导致陆地被淹没。
居民区受灾:部分房屋仍然可见，但许多看起来被水包围或部分淹没，这可能导致基础设施受损、居民被困或者财产损失。
树木和植被受影响:尽管树木仍然茂密，但被水淹没的情况可能导致植被根部受损，长期来看可能影响生态系统。
道路情况不明:由于洪水的覆盖，难以判断道路是否完好或者是否仍可通行，可能影响救援和疏散行动。"""
                # 模拟流式返回，将回复内容拆分成多个块
                chunk_size = 50
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    assistant_message.content += chunk
                    sse_chunk = {
                        "message_id": message_id,
                        "data": {
                            "content": chunk,
                            "done": i + chunk_size >= len(response_text)
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
                            "content": chunk,
                            "attachments": [],
                            "created_at": datetime.utcnow().isoformat()
                        })
                    )

                logger.info("流式回复生成完成")
                db.commit()
                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

            return StreamingResponse(
                sse_stream_generator(),
                media_type="text/event-stream"
            )
                
        elif text_content == "请判断受灾后A点到B点的道路是否通畅":
            async def sse_stream_generator() -> Generator[str, None, None]:
                # 调用大模型
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

                # 等待一下
                sleep(1)

                # 构造图片URL
                image_url = str(request_obj.url_for('static', path="test_image_3.png"))

                ## 给一个image_list可以拓展
                image_list = [image_url]
                
                for image in image_list:
                    image_id = generate_image_id(image_url)

                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image,'image_id':image_id,'type':"post"}})}\n\n"

                answer = "根据您所提供的图像，经过路径判断受灾后A点B点之间的道路受到灾害影响不通畅。"
                # Split answer into 1-2 byte chunks
                chunks = []
                i = 0
                while i < len(answer):
                    chunk_size = min(random.randint(1,2), len(answer)-i)
                    chunks.append(answer[i:i+chunk_size])
                    i += chunk_size
                
                stream_response = chunks
                
                for chunk in stream_response:
                    logger.info(f"原始 chunk: {chunk}")
                    content = chunk
                    if content.strip():
                        assistant_message.content += content
                        sse_chunk = {
                            "message_id": message_id,
                            "data": {
                                "content": content,
                                "done": False
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
            return StreamingResponse(
                sse_stream_generator(),
                media_type="text/event-stream"
            )

        elif text_content == "那受灾前A点到B点的道路是否通畅呢？":
            async def sse_stream_generator() -> Generator[str, None, None]:
                # 调用大模型
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

                # 等待一下
                sleep(1)

                # 构造图片URL
                image_url = str(request_obj.url_for('static', path="test_image_4.png"))

                ## 给一个image_list可以拓展
                image_list = [image_url]
                
                for image in image_list:
                    image_id = generate_image_id(image_url)
                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image,'image_id':image_id,'type':"post"}})}\n\n"

                answer = "根据您所提供的图像，经过路径判断受灾前A点B点之间的道路是通畅的。"
                # Split answer into 1-2 byte chunks
                chunks = []
                i = 0
                while i < len(answer):
                    chunk_size = min(random.randint(1,2), len(answer)-i)
                    chunks.append(answer[i:i+chunk_size])
                    i += chunk_size
                
                stream_response = chunks
                
                for chunk in stream_response:
                    logger.info(f"原始 chunk: {chunk}")
                    content = chunk
                    if content.strip():
                        assistant_message.content += content
                        sse_chunk = {
                            "message_id": message_id,
                            "data": {
                                "content": content,
                                "done": False
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
            return StreamingResponse(
                sse_stream_generator(),
                media_type="text/event-stream"
            )
        elif text_content == "请根据受灾场景综合判断房屋受损情况，要求尽可能的详细，且提供受灾图像的基本信息。":

            # 等待一下
            sleep(1)
                
            async def sse_stream_generator() -> Generator[str, None, None]:
                message_id = str(uuid.uuid4())
                assistant_message = ChatMessage(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content="",  # 后续流式填充内容
                    attachments=None
                )
                # 构造图片URL
                image_url = str(request_obj.url_for('static', path="test_image_5.png"))

                ## 给一个image_list可以拓展
                image_list = [image_url]
                    
                for image in image_list:
                    image_id = generate_image_id(image_url)
                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image,'image_id':image_id,'type':"post"}})}\n\n"


                db.add(assistant_message)
                response_text = """这张卫星图像展示了飓风过后一个受灾区域的全貌。图像中可以明显看到大片浑浊的水域覆盖了树林和部分居民区，表明该区域经历了严重洪水，陆地大面积被淹。部分房屋依然可辨，但许多建筑似乎被水包围或部分浸泡，暗示基础设施可能遭受破坏，居民也可能面临被困和财产损失的风险。虽然树木依旧茂密，但被淹的情况可能对植被的根系造成损伤，长期来看会影响生态系统；而由于洪水覆盖，难以判断道路的完好性和通行状况，这可能对救援和疏散行动构成阻碍。根据对36个建筑物的统计，数据显示有22个建筑物无损坏，主要分布在图像上半部分和右侧；4个建筑物显示轻微损坏，分散在图像中部和左侧；7个建筑物受严重损坏，主要集中在图像的中下部和左侧；另外还有4个建筑物未分类。这种分布表明，虽然大部分房屋没有明显损坏，但局部区域尤其是图像中下部和左侧，受灾情况较为严重，提示救援和恢复工作需针对性展开。"""
                # 模拟流式返回，将回复内容拆分成多个块
                chunk_size = 50
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    assistant_message.content += chunk
                    sse_chunk = {
                        "message_id": message_id,
                        "data": {
                            "content": chunk,
                            "done": i + chunk_size >= len(response_text)
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
                            "content": chunk,
                            "attachments": [],
                            "created_at": datetime.utcnow().isoformat()
                        })
                    )

                logger.info("流式回复生成完成")
                db.commit()
                yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

            return StreamingResponse(
                sse_stream_generator(),
                media_type="text/event-stream"
            )
                       
        else:
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

            return StreamingResponse(
                sse_stream_generator(),
                media_type="text/event-stream"
            )

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

    except HTTPException as e:
        db.rollback()  # 回滚事务
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )