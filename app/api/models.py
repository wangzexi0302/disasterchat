from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="发送给LLM的输入提示")
    model: Optional[str] = Field(None, description="要使用的模型（不指定则使用默认配置）")

class ChatResponse(BaseModel):
    response: str = Field(..., description="LLM的响应内容")

class StreamResponse(BaseModel):
    text: str = Field(..., description="LLM的流式响应片段")
    done: bool = Field(False, description="是否是最后一个片段")
