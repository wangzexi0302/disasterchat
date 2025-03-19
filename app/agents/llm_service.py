import ollama
from app.config import settings

def get_llm_response(prompt: str, model: str = None) -> str:
    """
    使用Ollama Python客户端获取LLM响应
    
    Args:
        prompt (str): 输入提示
        model (str, optional): 要使用的模型名称。默认为配置设置。
    
    Returns:
        str: LLM响应
    
    Raises:
        Exception: 如果与LLM服务通信出错
    """
    model = model or settings.default_model 
    
    try:
        # 使用ollama包直接调用模型
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    
    except Exception as e:
        raise Exception(f"Ollama客户端错误: {str(e)}")

def get_llm_stream_response(prompt: str, model: str = None):
    """
    使用Ollama Python客户端获取LLM流式响应
    
    Args:
        prompt (str): 输入提示
        model (str, optional): 要使用的模型名称。默认为配置设置。
    
    Yields:
        str: LLM响应片段
    
    Raises:
        Exception: 如果与LLM服务通信出错
    """
    model = model or settings.default_model
    
    try:
        # 使用ollama包的流式API
        for chunk in ollama.generate(model=model, prompt=prompt, stream=True):
            if 'response' in chunk:
                yield chunk['response']
    
    except Exception as e:
        raise Exception(f"Ollama客户端流式输出错误: {str(e)}")
