from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from app.api.models import ChatRequest, ChatResponse
from app.services.llm_service import get_llm_response, get_llm_stream_response

router = APIRouter(prefix="/api", tags=["llm"])

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    与LLM模型交互的端点
    """
    try:
        response = get_llm_response(request.prompt, request.model)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    与LLM模型交互的流式端点
    """
    try:
        async def generate():
            # 用于跟踪是否是最后一个响应
            is_last = False
            
            # 从生成器获取流式响应
            for text in get_llm_stream_response(request.prompt, request.model):
                is_last = False
                yield f"data: {json.dumps({'text': text, 'done': is_last})}\n\n"
            
            # 标记流式输出结束
            is_last = True
            yield f"data: {json.dumps({'text': '', 'done': is_last})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
