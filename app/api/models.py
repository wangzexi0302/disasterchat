from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Message(BaseModel):
    role: str = Field(..., description="消息角色，如user或assistant")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="消息历史记录")
    model: Optional[str] = Field(None, description="要使用的模型（不指定则使用默认配置）")

class FunctionCall(BaseModel):
    name: str = Field(..., description="调用的函数名称")
    arguments: str = Field(..., description="函数参数的JSON字符串")

class ToolCall(BaseModel):
    id: str = Field(..., description="工具调用ID")
    type: str = Field("function", description="工具类型")
    function: FunctionCall = Field(..., description="函数调用详情")

class AssistantMessage(BaseModel):
    content: Optional[str] = Field(None, description="消息内容")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="工具调用列表")

class ChatResponse(BaseModel):
    message: AssistantMessage = Field(..., description="助手的响应消息")
    
class StreamResponse(BaseModel):
    text: str = Field(..., description="LLM的流式响应片段")
    done: bool = Field(False, description="是否是最后一个片段")
