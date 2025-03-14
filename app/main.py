from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import uvicorn

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

# 移除API前缀，使路由直接在根路径下可访问
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
