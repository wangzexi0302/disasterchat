[
    {
        "name": "call_multimodel",
        "description": "根据用户要求对灾害影像进行粗略分析，可以分析灾前影像、灾后影像或同时分析两种影像",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "用户的具体问题或分析请求",
                    "required": True,
                },
                "pic_type": {
                    "type": "string",
                    "description": "要分析的影像类型：'pre'代表灾害前影像，'post'代表灾害后影像，'both'代表同时分析两种影像",
                    "enum": ["pre", "post", "both"],
                    "required": True,
                },
            },
            "required": ["message", "pic_type"],
        },
    },
    {
        "name": "call_qaagent",
        "description": "根据用户输入回答地灾相关的专业问题或一般性问题",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "用户的问题或查询内容",
                    "required": True,
                }
            },
            "required": ["message"],
        },
    },
    {
        "name": "call_image_analysis",
        "description": "根据用户要求对遥感影像进行处理（变化检测、语义分割），并调用路径计算、面积计算、数量计算工具，实现道路可达性估计、各类地物受灾面积估计、受灾建筑物损伤情况统计",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "用户的具体分析请求，包括要分析的内容和关注点",
                    "required": True,
                },
                "pic_type": {
                    "type": "string",
                    "description": "要分析的影像类型：'pre'代表灾害前影像，'post'代表灾害后影像，'both'代表需要同时分析两种影像(变化检测)",
                    "enum": ["pre", "post", "both"],
                    "required": True,
                },
            },
            "required": ["message", "pic_type"],
        },
    },
]
