# DisasterChat LLM后端

本项目集成FastAPI与Ollama，提供支持Agent和函数调用功能的LLM服务后端。

## 设置

1. 确保已安装并运行Ollama:
    https://ollama.com/download
    安装后需要运行Ollama，且拉取对应模型

    ```bash
    ollama pull llama3.2 #英文模型
    ollama pull qwen2.5 #中文
    ollama pull llava # 多模态模型
    ```

2. 使用Conda创建并激活虚拟环境:
   ```bash
   conda create -n disasterchat python=3.11
   conda activate disasterchat
   ```

3. 安装Poetry:
   ```bash
   pip install poetry
   ```

4. 使用Poetry安装项目依赖:
   ```bash
   # 在项目根目录下执行
   poetry install
   ```

## 功能特点

- **Agent形式的LLM交互**：模型作为Agent响应用户查询
- **函数调用**：支持工具函数调用，能够获取外部数据
- **流式输出**：提供流式API以实现实时响应
- **内置工具**：支持天气查询、灾害信息查询等功能

## 运行服务器

确保已激活conda环境 (disasterchat)，然后执行:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API端点

- **POST /chat**：聊天API，支持函数调用
- **POST /chat/stream**：流式聊天API
- **GET /health**：健康检查端点

## 测试

项目包含简单的API测试脚本，使用requests库直接发送请求：

1. 确保服务器已经启动且正在运行
2. 运行测试脚本:
   ```bash
   # 进入项目根目录
   python test/test_api.py
   ```

测试脚本将检查API的健康检查端点、聊天功能和函数调用功能，并显示响应结果。


## api todo

1. 流式传输 (SSE?)
2. redis
3. mysql
