from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import logging
from app.api.models import ChatRequest, ChatResponse, VLChatMessage
from app.agents.agent_service import AgentService
from app.agents.multimodal_agent import MultiModalAgent
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["llm"])

# 创建Agent服务实例
agent_service = AgentService()
multimodal_agent = MultiModalAgent()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    与LLM Agent交互的端点
    """
    try:
        model = request.model or settings.default_model
        logger.info(f"接收到聊天请求，使用模型：{model}")
        
        # 记录用户最新消息
        if request.messages:
            latest_msg = request.messages[-1].content
            logger.info(f"用户查询: {latest_msg[:100]}{'...' if len(latest_msg) > 100 else ''}")
        
        response = agent_service.run(
            messages=[msg.dict() for msg in request.messages], 
            model=model
        )
        
        logger.info("成功生成回复")
        return response
    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    与LLM Agent交互的流式端点
    """
    try:
        model = request.model or settings.default_model
        logger.info(f"接收到流式聊天请求，使用模型：{model}")
        
        # 记录用户最新消息
        if request.messages:
            latest_msg = request.messages[-1].content
            logger.info(f"用户流式查询: {latest_msg[:100]}{'...' if len(latest_msg) > 100 else ''}")
        
        async def generate():
            try:
                for chunk in agent_service.run_stream(
                    messages=[msg.dict() for msg in request.messages],
                    model=model
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # 标记流式输出结束
                logger.info("流式回复生成完成")
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error(f"流式生成过程中出错: {str(e)}", exc_info=True)
                # 在流中标记错误
                error_msg = {"error": str(e)}
                yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"流式聊天请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """健康检查端点"""
    logger.debug("接收到健康检查请求")
    return {"status": "ok"}
