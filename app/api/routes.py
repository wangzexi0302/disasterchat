import random
from time import sleep
import time
from datetime import datetime,timezone
from fastapi import APIRouter,Request,Depends,Form, BackgroundTasks, HTTPException,UploadFile,File,status
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.database.db_setup import get_db
from app.api.database.models import DBSession, ChatMessage, Image, MessageImage, PromptTemplate
from pydantic import BaseModel
from datetime import datetime  # 添加此行
from typing import Generator, List, Dict, Any
import redis
import json
from sqlalchemy import orm
from sqlalchemy.exc import SQLAlchemyError
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
from sqlalchemy import select,delete,desc


logger = logging.getLogger(__name__)  # 使用模块级 logger

# 异步锁，用于避免资源竞争
response_lock = asyncio.Lock()

# Base.metadata.create_all(bind=engine)

# 删除所有表（生产环境禁用！）
# Base.metadata.drop_all(engine)

# 重新创建表
# Base.metadata.create_all(engine)

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
    type: str      # 类型（如pre/jpeg）

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




# 核心接口实现
#开启新会话
@router.post("/chat/new_session", response_model=NewSessionResponse)
async def new_session(db: AsyncSession = Depends(get_db)):
    try:
        session_id = str(uuid.uuid4())

        # 定义三个指定的字符串
        session_name_options = ["地震后受灾情况", "洪水受灾情况", "飓风过境影像分析"]
        # 随机选择一个字符串作为会话名称
        session_name = "新会话"

        # 存入 MySQL
        db_session = DBSession(
            id=session_id,
            name=session_name
        )
        db.add(db_session)
        # 异步提交数据库事务
        await db.commit()

        return {"session_id": session_id}
    except Exception as e:
        # 异步回滚数据库事务
        await db.rollback()
        raise HTTPException(500, detail=f"数据库写入失败: {str(e)}")
    finally:
        # 异步关闭数据库会话
        await db.close()

# 修改会话名称接口
@router.post("/chat/update_session_name", response_model=NewSessionResponse)
async def update_session_name(
    session_id: str = Form(...),
    new_name: str = Form(...),
    db: AsyncSession = Depends(get_db)  # 使用异步会话
):
    try:
        # 异步查询数据库
        result = await db.execute(
            select(DBSession).filter(DBSession.id == session_id)
        )
        db_session = result.scalar_one_or_none()

        if not db_session:
            logging.error(f"会话ID不存在: {session_id}")
            raise HTTPException(400, "Invalid session ID")

        # 更新会话名称
        db_session.name = new_name
        
        # 异步提交事务
        await db.commit()

        return {"session_id": session_id}

    except HTTPException as e:
        # 已有显式异常，直接抛出
        raise e
    except Exception as e:
        # 其他异常回滚事务
        await db.rollback()
        logging.error(f"更新会话名称失败: {str(e)}")
        raise HTTPException(500, detail=f"更新会话名称失败: {str(e)}")
    finally:
        # 异步关闭会话（可选，由FastAPI自动管理）
        await db.close()

# 删除会话接口
@router.post("/chat/delete_session", response_model=NewSessionResponse)
async def delete_session(
    session_id: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    logging.info(f"收到删除会话的请求。会话 ID: {session_id}")
    try:
        # 异步验证数据库会话
        result = await db.execute(
            select(DBSession).filter(DBSession.id == session_id)
        )
        db_session = result.scalar_one_or_none()
        if not db_session:
            logging.error(f"会话id不存在: {session_id}")
            raise HTTPException(400, "Invalid session ID in database")

        # 先通过子查询找到要删除的 MessageImage 记录
        subquery = select(ChatMessage.id).where(ChatMessage.session_id == session_id).subquery()
        await db.execute(
            delete(MessageImage).where(MessageImage.message_id.in_(subquery))
        )

        # 再删除 chat_messages 表中关联的记录
        await db.execute(
            delete(ChatMessage).where(ChatMessage.session_id == session_id)
        )

        # 异步删除数据库中的会话记录
        await db.delete(db_session)
        await db.commit()
        logging.info(f"Deleted session {session_id} from database.")

        return {"session_id": session_id}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        await db.rollback()
        logging.error(f"操作出错: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"操作出错: {str(e)}")
    finally:
        await db.close()#查询历史会话接口
@router.post("/chat/history_list", response_model=HistoryListResponse)
async def get_history_list(db: AsyncSession = Depends(get_db)):
    """
    获取聊天历史列表的接口函数
    :param db: 数据库会话
    :return: 包含历史会话数据的响应
    """
    # 定义缓存键
    cache_key = "history_sessions"
    # 定义数据库查询，按创建时间降序排序
    query = select(DBSession).order_by(desc(DBSession.created_at))
    # 缓存时间，单位为秒
    cache_time = 3600

    try:
        # 异步执行查询获取数据库数据
        result = await db.execute(query)
        db_data = result.scalars().all()
        # 处理数据库数据，将其转换为所需的字典格式
        data = [
            {"sessionId": s.id, "name": s.name, "content": ""} if isinstance(s, DBSession) else
            {"id": p.id, "name": p.name, "content": p.content} if isinstance(p, PromptTemplate) else
            {} for s in db_data
        ]

        return {"data": {"templates": data}}
    except Exception as e:
        logging.error(f"查询历史会话列表时出错: {str(e)}")
        # 这里可以根据具体情况返回合适的错误响应
        return {"data": {"templates": []}, "error": str(e)}

    

#历史消息详情接口
@router.post("/chat/history_detail")
async def get_history_detail(
    request_obj: Request,
    request: HistoryDetailRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        # 异步查询
        query = select(ChatMessage).filter(
            ChatMessage.session_id == request.sessionId
        ).order_by(ChatMessage.created_at.asc())
        result = await db.execute(query)
        messages = result.scalars().all()

        response_messages = []
        for msg in messages:
            # 异步查询关联图片
            image_query = select(Image).join(MessageImage).filter(
                MessageImage.message_id == msg.id
            )
            images = await db.execute(image_query)
            images = images.scalars().all()

            attachments = [
                ImageAttachment(
                    url=f"{request_obj.base_url}{img.file_path.replace('\\', '/')}",
                    type=img.type
                ) for img in images
            ]
            response_messages.append(ChatMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                attachments=attachments,
                created_at=msg.created_at.isoformat()
            ))

        for msg in response_messages[:10]:  # 输出前3条消息
            logger.info(f"消息ID: {msg.id}, 角色: {msg.role}, 内容: {msg.content[:50]}...")

        return HistoryDetailResponse(data={"messages": response_messages})

    except Exception as e:
        logger.error(f"历史记录查询错误: {str(e)}")
        raise HTTPException(500, detail="服务器内部错误")
#图片上传接口
@router.post("/chat/upload_image", response_model=UploadImageResponse)
async def upload_image(type: str = Form(...), files: list[UploadFile] = File(...), db: AsyncSession = Depends(get_db)):
    if not files:
        raise HTTPException(400, "未提供图片")
    image_ids = []
    upload_dir = "uploads"
    # 检查上传目录是否存在，不存在则创建
    if not os.path.exists(upload_dir):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, os.makedirs, upload_dir)

    for file in files:
        try:
            # 生成保存路径
            file_extension = file.filename.split('.')[-1]
            save_path = os.path.join(upload_dir, f"{uuid.uuid4()}.{file_extension}")
            # 异步写入文件
            loop = asyncio.get_running_loop()
            file_content = await file.read()
            await loop.run_in_executor(None, lambda: open(save_path, "wb").write(file_content))
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
            await db.rollback()
            raise HTTPException(500, f"上传图片时发生错误: {str(e)}")

    try:
        # 异步提交数据库事务
        await db.commit()
    except Exception as e:
        # 数据库提交失败时进行回滚操作
        await db.rollback()
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


# 对话暂停接口
@router.post("/chat/pause")
async def pause_session(
    session_id: str = Form(...)
):
    try:
        # 在 Redis 中标记会话需要暂停
        redis_client.set(f"pause_flag:{session_id}","1",5)
        logger.info(f"会话 {session_id} 收到暂停指令，暂停请求已发送")
        return {"status": "success", "message": "已发送暂停请求，当前会话请求将逐步终止"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停操作失败: {str(e)}"
        )
    

# 发送消息接口
@router.post("/chat/send")
async def send_message(
    request: SendMessageRequest,
    request_obj: Request,
    db: AsyncSession = Depends(get_db)
) -> StreamingResponse:
    try:
        session_id = request.sessionId

        # 解析消息内容
        text_content = ""
        image_ids = []
        is_multimodal = False

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
            text_content = text_content.strip()
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid content type"
            )

        #打印图片
        logger.info(f"Received message: {text_content}{image_ids}")

        # 数据库事务处理
        async with db.begin() as transaction:
            user_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=request.message.role,
                content=text_content,
                attachments=json.dumps(image_ids) if image_ids else None
            )
            db.add(user_message)
            await db.flush()


            # 验证并关联图片
            image_paths = []
            for img_id in image_ids:
                result = await db.execute(select(Image).filter(Image.id == img_id))
                db_image = result.scalar_one_or_none()
                if not db_image:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Image not found: {img_id}"
                    )
                image_paths.append(db_image.file_path)  # 收集图片路径

                # 消息-图片关联
                db.add(MessageImage(
                    message_id=user_message.id,
                    image_id=img_id
                ))
            await db.commit()
            logging.info(f"成功插入消息: {user_message.id}, 关联图片数量: {len(image_ids)}")    


        # 构造LLM消息格式
        llm_messages = []
        if is_multimodal:
            llm_messages = [{
                "role": request.message.role,
                "content": [
                    {"type": "text", "text": text_content},
                    *[{"type": "image", "image_data": path} for path in image_paths]
                ]
            }]
            logger.info(f"多模态 message: {llm_messages}")
        else:
            llm_messages = [{"role": request.message.role, "content": text_content}]

        # 特定场景处理
        if text_content == "变化检测":
            async def scene_sse_generator():
                try:
                    # 使用独立事务创建AI消息
                    async with db.begin() as new_transaction:
                        message_id = str(uuid.uuid4())
                        assistant_message = ChatMessage(
                            id=message_id,
                            session_id=session_id,
                            role="assistant",
                            content=""
                        )
                        db.add(assistant_message)
                        await db.commit()  # 立即提交事务

                    # 转换同步生成器为异步生成器
                    stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")
                    async_gen = async_generator_wrapper(stream_response)

                    image_url = str(request_obj.url_for('static', path="test_image_2.png"))
                    image_list = [image_url]

                    # 发送图片
                    for image in image_list:
                        image_id = generate_image_id(image_url)
                        yield f"data: {json.dumps({
                            'message_id': message_id,
                            'data': {
                                'done': False,
                                'image_url': image,
                                'image_id': image_id,
                                'type': 'post'
                            }
                        })}\n\n"

                    # 处理LLM响应
                    async for chunk in async_gen:
                        content = (
                            chunk.get("message", {}).get("content", "")  # Ollama 格式
                        )
                        if not content:
                            content = chunk.get("text", "") or chunk.get("output", "") or ""
                        if content.strip():
                            # 使用独立事务更新内容
                            async with db.begin() as update_transaction:
                                assistant_message.content += content
                                await db.commit()  # 立即提交事务

                            yield f"data: {json.dumps({
                                'message_id': message_id,
                                'data': {
                                    'content': content,
                                    'done': chunk.get('done', False)
                                }
                            })}\n\n"

                    # 最终提交事务
                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

                except Exception as e:
                    await db.rollback()
                    yield f"data: {json.dumps({'error': str(e), 'detail': 'Inference failed'})}\n\n"

            return StreamingResponse(
                scene_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        elif text_content == "请告诉我灾后影像的大致受灾情况":
            async def scene_sse_generator():
                await asyncio.sleep(7)
                try:
                    message_id = str(uuid.uuid4())

                    # 使用独立事务创建AI消息
                    async with db.begin() as new_transaction:
                        assistant_message = ChatMessage(
                            id=message_id,
                            session_id=session_id,
                            role="assistant",
                            content=""
                        )
                        db.add(assistant_message)
                        await db.commit()  # 提交事务

                    response_text = """这张影像显示了一个受灾区域，飓风后的卫星图像。从图像来看，主要的受灾情况包括:
大面积积水:图像显示大片的浑浊水域，覆盖了树林和部分居民区。这表明该区域可能经历了严重的洪水，导致陆地被淹没。
居民区受灾:部分房屋仍然可见，但许多看起来被水包围或部分淹没，这可能导致基础设施受损、居民被困或者财产损失。
树木和植被受影响:尽管树木仍然茂密，但被水淹没的情况可能导致植被根部受损，长期来看可能影响生态系统。
道路情况不明:由于洪水的覆盖，难以判断道路是否完好或者是否仍可通行，可能影响救援和疏散行动。"""


                    chunk_size = 5
                    for i in range(0, len(response_text), chunk_size):
                        chunk = response_text[i:i+chunk_size]

                        # 使用独立事务更新内容
                        async with db.begin() as update_transaction:
                            assistant_message.content += chunk
                            await db.commit()

                        yield f"data: {json.dumps({
                            'message_id': message_id,
                            'data': {
                                'content': chunk,
                                'done': i+chunk_size >= len(response_text)
                            }
                        })}\n\n"
                        await asyncio.sleep(1)

                    # 最终提交事务
                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"

                except Exception as e:
                    await db.rollback()
                    yield f"data: {json.dumps({'error': str(e), 'detail': 'Inference failed'})}\n\n"

            return StreamingResponse(scene_sse_generator(), media_type="text/event-stream")
        
        elif text_content == "请判断受灾后A点到B点的道路是否通畅":
            async def scene_sse_generator():
                await asyncio.sleep(13)
                try:
                    # 使用独立事务创建AI消息
                    async with db.begin() as new_transaction:
                        message_id = str(uuid.uuid4())
                        assistant_message = ChatMessage(
                            id=message_id,
                            session_id=session_id,
                            role="assistant",
                            content=""
                        )
                        db.add(assistant_message)
                        await db.commit()

                    answer = "根据您所提供的图像，经过路径判断受灾后A点B点之间的道路受到灾害影响不通畅。"
                    chunks = [answer[i:i + 5] for i in range(0, len(answer), 5)]

                    # 处理文本响应
                    for chunk in chunks:
                        # 使用独立事务更新内容
                        async with db.begin() as update_transaction:
                            assistant_message.content += chunk
                            await db.commit()
                        yield f"data: {json.dumps({
                            'message_id': message_id,
                            'data': {
                                'content': chunk,
                                'done': False
                            }
                        })}\n\n"
                        await asyncio.sleep(1)

                    image_url = str(request_obj.url_for('static', path="test_image_3.png"))
                    image_list = [image_url]

                    # 发送图片
                    for image in image_list:
                        image_id = generate_image_id(image_url)
                        yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image, 'image_id': image_id, 'type': 'post'}})}\n\n"                    

                    # 最终提交事务
                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"
                except Exception as e:
                    await db.rollback()
                    yield f"data: {json.dumps({'error': str(e), 'detail': 'Inference failed'})}\n\n"

            return StreamingResponse(
                scene_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )

        elif text_content == "那受灾前A点到B点的道路是否通畅呢？":
            async def scene_sse_generator():
                await asyncio.sleep(12)
                try:
                    # 使用独立事务创建AI消息
                    async with db.begin() as new_transaction:
                        message_id = str(uuid.uuid4())
                        assistant_message = ChatMessage(
                            id=message_id,
                            session_id=session_id,
                            role="assistant",
                            content=""
                        )
                        db.add(assistant_message)
                        await db.commit()


                    answer = "根据您所提供的图像，经过路径判断受灾前A点B点之间的道路是通畅的。"
                    chunks = [answer[i:i + 5] for i in range(0, len(answer), 5)]

                    # 处理文本响应
                    for chunk in chunks:
                        # 使用独立事务更新内容
                        async with db.begin() as update_transaction:
                            assistant_message.content += chunk
                            await db.commit()
                        yield f"data: {json.dumps({
                            'message_id': message_id,
                            'data': {
                                'content': chunk,
                                'done': False
                            }
                        })}\n\n"
                        await asyncio.sleep(1)

                    image_url = str(request_obj.url_for('static', path="test_image_4.png"))
                    image_list = [image_url]

                    # 发送图片
                    for image in image_list:
                        image_id = generate_image_id(image_url)
                        yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image, 'image_id': image_id, 'type': 'post'}})}\n\n"

                    # 最终提交事务
                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"
                except Exception as e:
                    await db.rollback()
                    yield f"data: {json.dumps({'error': str(e), 'detail': 'Inference failed'})}\n\n"

            return StreamingResponse(
                scene_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )

        elif text_content == "请根据受灾场景综合判断房屋受损情况，要求尽可能的详细，且提供受灾图像的基本信息。":
            async def scene_sse_generator():
                await asyncio.sleep(20)
                try:
                    # 使用独立事务创建AI消息
                    async with db.begin() as new_transaction:
                        message_id = str(uuid.uuid4())
                        assistant_message = ChatMessage(
                            id=message_id,
                            session_id=session_id,
                            role="assistant",
                            content=""
                        )
                        db.add(assistant_message)
                        await db.commit()


                    response_text = """这张卫星图像展示了飓风过后一个受灾区域的全貌。图像中可以明显看到大片浑浊的水域覆盖了树林和部分居民区，表明该区域经历了严重洪水，陆地大面积被淹。部分房屋依然可辨，但许多建筑似乎被水包围或部分浸泡，暗示基础设施可能遭受破坏，居民也可能面临被困和财产损失的风险。虽然树木依旧茂密，但被淹的情况可能对植被的根系造成损伤，长期来看会影响生态系统；而由于洪水覆盖，难以判断道路的完好性和通行状况，这可能对救援和疏散行动构成阻碍。根据对36个建筑物的统计，数据显示有22个建筑物无损坏，主要分布在图像下半部分和右侧；4个建筑物显示轻微损坏，分散在图像右下部；7个建筑物受严重损坏，主要集中在图像的中部；另外还有4个建筑物无法识别。这种分布表明，虽然大部分房屋没有明显损坏，但局部区域尤其是图像中下部和左侧，受灾情况较为严重，提示救援和恢复工作需针对性展开。"""

                    chunk_size = 5
                    for i in range(0, len(response_text), chunk_size):
                        chunk = response_text[i:i + chunk_size]
                        # 使用独立事务更新内容
                        async with db.begin() as update_transaction:
                            assistant_message.content += chunk
                            await db.commit()
                        yield f"data: {json.dumps({
                            'message_id': message_id,
                            'data': {
                                'content': chunk,
                                'done': i + chunk_size >= len(response_text)
                            }
                        })}\n\n"
                        await asyncio.sleep(1)

                    image_url = str(request_obj.url_for('static', path="test_image_5.png"))
                    image_list = [image_url]

                    # 发送图片
                    for image in image_list:
                        image_id = generate_image_id(image_url)
                        yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image, 'image_id': image_id, 'type': 'post'}})}\n\n"

                    # 最终提交事务
                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"
                except Exception as e:
                    await db.rollback()
                    yield f"data: {json.dumps({'error': str(e), 'detail': 'Inference failed'})}\n\n"

            return StreamingResponse(
                scene_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )

        elif text_content == "请详细介绍灾害后各地物的面积统计":
            
            async def scene_sse_generator():
                await asyncio.sleep(7)
                try:
                    # 使用独立事务创建AI消息
                    async with db.begin() as new_transaction:
                        message_id = str(uuid.uuid4())
                        assistant_message = ChatMessage(
                            id=message_id,
                            session_id=session_id,
                            role="assistant",
                            content=""
                        )
                        db.add(assistant_message)
                        await db.commit()

                    response_text = """根据灾后影像的分割结果，各个地物的面积如下： \n1. 建筑物：66260.0 平方米。 \n2. 水体：546520.0 平方米。 \n3. 道路：118720.0平方米"""

                    chunk_size = 5
                    for i in range(0, len(response_text), chunk_size):
                        chunk = response_text[i:i + chunk_size]
                        # 使用独立事务更新内容
                        async with db.begin() as update_transaction:
                            assistant_message.content += chunk
                            await db.commit()
                        yield f"data: {json.dumps({
                            'message_id': message_id,
                            'data': {
                                'content': chunk,
                                'done': i + chunk_size >= len(response_text)
                            }
                        })}\n\n"
                        await asyncio.sleep(1)
                    image_url = str(request_obj.url_for('static', path="test_image_3.png"))
                    image_list = [image_url]

                    # 发送图片
                    for image in image_list:
                        image_id = generate_image_id(image_url)
                        yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image, 'image_id': image_id, 'type': 'post'}})}\n\n"


                    # 最终提交事务
                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"
                except Exception as e:
                    await db.rollback()
                    yield f"data: {json.dumps({'error': str(e), 'detail': 'Inference failed'})}\n\n"

            return StreamingResponse(
                scene_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )

        elif text_content == "请在灾害前影像中规划A、B两点的具体通行路径":
            async def scene_sse_generator():
                await asyncio.sleep(10)
                try:
                    # 使用独立事务创建AI消息
                    async with db.begin() as new_transaction:
                        message_id = str(uuid.uuid4())
                        assistant_message = ChatMessage(
                            id=message_id,
                            session_id=session_id,
                            role="assistant",
                            content=""
                        )
                        db.add(assistant_message)
                        await db.commit()

                    answer = "根据您所提供的图像，A、B两点间的具体路径如下："
                    chunks = [answer[i:i + 5] for i in range(0, len(answer), 5)]

                    # 处理文本响应
                    for chunk in chunks:
                        # 使用独立事务更新内容
                        async with db.begin() as update_transaction:
                            assistant_message.content += chunk
                            await db.commit()
                        yield f"data: {json.dumps({
                            'message_id': message_id,
                            'data': {
                                'content': chunk,
                                'done': False
                            }
                        })}\n\n"
                        await asyncio.sleep(1)

                    image_list = [
                        str(request_obj.url_for('static', path="path_pic.jpg")),
                        str(request_obj.url_for('static', path="path_mask.jpg"))
                    ]

                    # 发送图片
                    for image in image_list:
                        image_id = generate_image_id(image)
                        yield f"data: {json.dumps({'message_id': message_id, 'data': {'done': False, 'image_url': image, 'image_id': image_id, 'type': 'post'}})}\n\n"


                    # 最终提交事务
                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    yield f"data: {json.dumps({'message_id': message_id, 'data': {'content': '', 'done': True}})}\n\n"
                except Exception as e:
                    await db.rollback()
                    yield f"data: {json.dumps({'error': str(e), 'detail': 'Inference failed'})}\n\n"

            return StreamingResponse(
                scene_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )

        # 默认处理逻辑
        async def sse_stream_generator() -> Generator[str, None, None]:
            logger.info("默认处理逻辑")
            try:
                async with response_lock:
                    # 创建AI消息记录
                    pause_flag_key = f"pause_flag:{session_id}"  # 定义暂停标志键
                    
                    assistant_message = ChatMessage(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        role="assistant",
                        content="",  # 后续流式填充内容
                        attachments=None,
                    )

                    async with db.begin() as sub_transaction:
                        db.add(assistant_message)
                        await db.commit()

                    # 调用大模型（假设为异步调用）
                    if is_multimodal:
                        stream_response = multimodal_agent.run_stream(llm_messages, model="llava:latest")
                    else:
                        stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")

                    # 如果是同步生成器，包装为异步生成器
                    if not hasattr(stream_response, "__aiter__"):
                        stream_response = async_generator_wrapper(stream_response)

                    async for chunk in stream_response:

                        if redis_client.get(pause_flag_key):
                            logger.info(f"会话 {session_id} 收到暂停指令，终止流式回复")
                            break

                        content = (
                            chunk.get("message", {}).get("content", "")  # Ollama 格式
                        )
                        if not content:
                            content = chunk.get("text", "") or chunk.get("output", "") or ""
                        if content.strip():
                            async with db.begin() as update_transaction:
                                assistant_message.content += content
                                await db.commit()

                            sse_chunk = {
                                "message_id": assistant_message.id,
                                "data": {
                                    "content": content,
                                    "done": chunk.get("done", False)
                                }
                            }
                            # 发送SSE消息
                            yield f"data: {json.dumps(sse_chunk)}\n\n"

                    async with db.begin() as final_transaction:
                        assistant_message.created_at = datetime.now(timezone.utc)
                        await db.commit()

                    logger.info("流式回复生成完成")
                    yield f"data: {json.dumps({'message_id': assistant_message.id, 'data': {'content': '', 'done': True}})}\n\n"

            except Exception as e:
                await db.rollback()
                error_chunk = {
                    "error": str(e),
                    "detail": "Inference failed"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            sse_stream_generator(),
            media_type="text/event-stream"
        )


    except HTTPException as e:
        await db.rollback()
        raise e
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )
# 将同步生成器包装为异步生成器
async def async_generator_wrapper(sync_gen):
    for chunk in sync_gen:
        yield chunk
        await asyncio.sleep(0)  # 模拟异步行为，避免阻塞事件循环


#总结标题接口
@router.get("/session/{session_id}/title_summary")
async def get_session_title_summary(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        # 查询会话中的第一条用户消息
        first_user_message = await db.execute(
            select(ChatMessage).where(
                ChatMessage.session_id == session_id, ChatMessage.role == "user"
            ).order_by(ChatMessage.created_at).limit(1)
        )
        first_user_message = first_user_message.scalar_one_or_none()

        if not first_user_message:
            return {"summary": "未找到对话记录"}

        # 查询对应的第一条 AI 回复消息
        first_ai_message = await db.execute(
            select(ChatMessage).where(
                ChatMessage.session_id == session_id, ChatMessage.role == "assistant",
                ChatMessage.created_at > first_user_message.created_at
            ).order_by(ChatMessage.created_at).limit(1)
        )
        first_ai_message = first_ai_message.scalar_one_or_none()

        if not first_ai_message:
            # 如果没有找到对应的 AI 回复，只使用用户消息进行总结
            combined_content = first_user_message.content
        else:
            # 组合用户消息和 AI 回复的内容
            combined_content = f"{first_user_message.content} {first_ai_message.content}"

        # 构造发送给 AI 的消息
        llm_messages = [{"role": "user", "content": combined_content + " 十个字内总结，总结成一个提问类型的标题"}]

        # 调用 agent 服务获取总结结果

        stream_response = agent_service.run_stream(llm_messages, model="qwen2:7b")

        # 如果是同步生成器，包装为异步生成器
        if not hasattr(stream_response, "__aiter__"):
            stream_response = async_generator_wrapper(stream_response)

        summary = ""
        async for chunk in stream_response:
            logger.info(f"原始 chunk: {chunk}")
            content = (
                    chunk.get("message", {}).get("content", "")  # Ollama 格式
                        )
            if not content:
                content = chunk.get("text", "") or chunk.get("output", "") or ""
            summary += content

        # 确保总结在十个字之内
        summary = summary[1:11]

        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}