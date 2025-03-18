from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

# 声明基类
Base = declarative_base()

class DBSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=DateTime.utcnow)