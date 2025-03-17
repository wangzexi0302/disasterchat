from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from app.api.models import ChatRequest, ChatResponse
from app.agents.agent_service import AgentService
from app.config import settings

router = APIRouter(tags=["llm"])

# 创建Agent服务实例
agent_service = AgentService()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    与LLM Agent交互的端点
    """
    try:
        model = request.model or settings.default_model
        response = agent_service.run(
            messages=[msg.dict() for msg in request.messages], 
            model=model
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    与LLM Agent交互的流式端点
    """
    try:
        model = request.model or settings.default_model
        
        async def generate():
            for chunk in agent_service.run_stream(
                messages=[msg.dict() for msg in request.messages],
                model=model
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # 标记流式输出结束
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}
