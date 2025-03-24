from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging


# 数据库连接 URL
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://disasterchat_user:123456@localhost/disasterchat"

# 创建数据库引擎
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 示例：如果需要进一步调整日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.getLogger('sqlalchemy.engine').addHandler(handler)
def get_db():
    """获取数据库会话的依赖函数"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()