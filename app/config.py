import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # 应用配置
    app_name: str = "DisasterChat"
    debug: bool = True
    
    # API 配置
    api_prefix: str = "/api"
    
    # Ollama 配置
    ollama_base_url: str = "http://localhost:11434"  # 仍然保留，以便自定义Ollama服务地址
    default_model: str = "qwen2.5"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 设置Ollama客户端的主机
        # 从ollama_base_url中提取主机和端口信息
        if self.ollama_base_url:
            os.environ["OLLAMA_HOST"] = self.ollama_base_url

settings = Settings()
