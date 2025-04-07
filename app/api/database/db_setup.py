from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import logging

# 数据库连接 URL（修改驱动为异步）
SQLALCHEMY_DATABASE_URL = "mysql+aiomysql://disasterchat_user:123456@localhost/disasterchat"

# 创建异步数据库引擎
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,  # 初始连接数
    max_overflow=10,  # 最大溢出连接数
    pool_timeout=30  # 连接获取超时时间
)

# 创建异步会话工厂
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 配置数据库日志（可选）
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.getLogger('sqlalchemy.engine').addHandler(handler)

async def get_db():
    """获取异步数据库会话的依赖函数"""
    async with AsyncSessionLocal() as session:
        yield session