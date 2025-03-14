from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class Tool(ABC):
    """工具基类，所有自定义工具都应继承此类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """工具参数定义 (OpenAI函数调用格式)"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行工具功能"""
        pass
    
    def to_function_definition(self) -> Dict[str, Any]:
        """转换为OpenAI函数定义格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": [k for k, v in self.parameters.items() 
                            if v.get("required", False)]
            }
        }
