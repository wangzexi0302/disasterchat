from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import uvicorn
import logging
import os
from datetime import datetime

# 配置日志
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DisasterChat Agent API",
    description="FastAPI后端与Ollama集成提供LLM Agent服务，支持函数调用",
    version="0.1.0"
)

# 添加CORS中间件，允许流式响应
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("DisasterChat API服务启动中...")

# 移除API前缀，使路由直接在根路径下可访问
app.include_router(router)

if __name__ == "__main__":
    logger.info("以独立模式启动服务器")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
