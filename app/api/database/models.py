from sqlalchemy import Column, String, DateTime, ForeignKey,Integer
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime,timezone
import uuid

# 声明基类
Base = declarative_base()


# ======================
# 1. 会话表（核心）
# ======================
class DBSession(Base):
    """存储会话元数据"""
    __tablename__ = "chat_sessions"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))  # UUID格式会话ID
    name = Column(String(255), nullable=False)  # 会话名称（如"会话1"）
    created_at = Column(DateTime, default=datetime.utcnow)  # 创建时间
    
    # 关系：一个会话包含多条消息
    messages = relationship("ChatMessage", back_populates="session")


# ======================
# 2. 消息表（核心）
# ======================
class ChatMessage(Base):
    """存储聊天消息内容"""
    __tablename__ = "chat_messages"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))  # 消息ID
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), index=True)  # 所属会话ID
    role = Column(String(20), nullable=False)  # 消息角色：user/assistant
    content = Column(String(2000))  # 文本内容
    attachments = Column(String(255))  # 存储图片ID列表（JSON格式，如["img_1", "img_2"]）
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))  # 时区感知时间戳
    # 关系：消息属于某个会话
    session = relationship("DBSession", back_populates="messages")
    
    # 关系：消息关联的图片（通过中间表）
    image_relations = relationship("MessageImage", back_populates="message")


# ======================
# 3. 图片表（独立资源）
# ======================
class Image(Base):
    """存储图片元数据及本地路径"""
    __tablename__ = "images"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4())) # 图片ID
    file_path = Column(String(255), nullable=False)  # 本地存储路径（如"uploads/1.jpg"）
    file_type = Column(String(50), nullable=False)  # 图片类型（如image/jpeg）
    uploaded_at = Column(DateTime, default=datetime.utcnow)  # 上传时间
    type = Column(String(50))  # 新增字段type
    
    # 关系：图片关联的消息（通过中间表）
    message_relations = relationship("MessageImage", back_populates="image")


# ======================
# 4. 消息-图片关联表（中间表）
# ======================
class MessageImage(Base):
    """多对多关系：一条消息可关联多张图片，一张图片可被多条消息引用"""
    __tablename__ = "message_images"
    
    message_id = Column(String(36), ForeignKey("chat_messages.id"), primary_key=True)
    image_id = Column(String(36), ForeignKey("images.id"), primary_key=True)
    
    # 关系：关联消息
    message = relationship("ChatMessage", back_populates="image_relations")
    
    # 关系：关联图片
    image = relationship("Image", back_populates="message_relations")

class PromptTemplate(Base):
    __tablename__ = "prompt_templates"
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    content = Column(String(2000), nullable=False)