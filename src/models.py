# Date    : 2024/7/17 15:08
# File    : models.py
# Desc    : 查询改写API的数据模型
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Optional, Dict, Union, Any
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# 导入设置模块
from src.settings import settings
from src.constants import QueryType


# 大模型请求消息结构体
class Message(BaseModel):
    role: str = Field(..., description="消息的角色，枚举值为 'user' 或 'assistant' 或 'system'")
    content: str = Field(..., description="消息的实际内容，文本大模型的类型为 String，多模态大模型的类型为 List<VLContent>")
    reasoningContent: Optional[str] = Field(None, description="消息思考过程的实际内容")

    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v

    @validator('content')
    def validate_content(cls, v):
        if not v or v.isspace():
            raise ValueError("Content must not be empty or only whitespace")
        return v


# 信息抽取规则模型
class ExtractionRule(BaseModel):
    desc: str = Field(..., description="抽取信息的描述")
    valueType: str = Field(..., description="抽取信息的值类型，如int、string等")
    keyName: str = Field(..., description="抽取信息的键名")
    defaultValue: Any = Field(None, description="抽取信息的默认值")


# 查询改写请求模型
class QueryRewriteRequest(BaseModel):
    requestId: str
    query: str
    historyList: Optional[List[Message]] = []
    enableExtraction: Optional[bool] = False
    extractionRules: Optional[List[ExtractionRule]] = None

    @validator('query')
    def validate_query(cls, v):
        if not v or v.isspace():
            raise ValueError("Query must not be empty or only whitespace")
        if len(v) > 128:
            raise ValueError("Query length must not exceed 128 characters")
        return v

    @validator('historyList')
    def validate_history(cls, v):
        if not v:
            return v
        
        # 按Q&A对来限制，最多保留5对Q&A（即最多10个Message）
        # 从后往前找，保留完整的Q&A对
        max_pairs = 5
        result = []
        current_pairs = 0
        
        # 从后往前遍历
        i = len(v) - 1
        while i >= 0 and current_pairs < max_pairs:
            msg = v[i]
            
            if msg.role == "assistant":
                # 找到assistant消息，添加到结果开头
                result.insert(0, msg)
                
                # 查找前面是否有对应的user消息
                if i > 0 and v[i - 1].role == "user":
                    result.insert(0, v[i - 1])
                    current_pairs += 1
                    i -= 2  # 跳过这个user消息
                else:
                    # 单独的assistant消息也算一对
                    current_pairs += 1
                    i -= 1
            elif msg.role == "user":
                # 单独的user消息（没有对应的assistant回答）
                result.insert(0, msg)
                current_pairs += 1
                i -= 1
            else:
                # 其他角色的消息，跳过
                i -= 1
        
        return result
    
    @validator('enableExtraction')
    def validate_enable_extraction(cls, v):
        # 兼容Java上游可能传入null的情况
        if v is None:
            return False
        return v


# 查询改写结果模型
class QueryRewriteResults(BaseModel):
    requestId: str
    result: str
    needRewrite: bool
    processTime: int
    isSummary: Optional[bool] = None  # 是否为跨文档总结类型查询
    keywords: Optional[List[str]] = None  # 提取的关键词列表
    queryType: Optional[int] = None  # 查询类型：0-其他类，1-总结类，2-建议类，3-发言稿
    extractedInfo: Optional[List[Dict[str, Any]]] = None  # 提取的信息列表

    # 根据settings设置决定是否包含summary相关字段
    @model_validator(mode='after')
    def check_summary_fields(self) -> 'QueryRewriteResults':
        if not settings.ENABLE_SUMMARY_CHECK:
            self.__dict__.pop('isSummary', None)
            self.__dict__.pop('keywords', None)
            
        # 处理查询分类字段
        if not settings.ENABLE_QUERY_CLASSIFY:
            self.__dict__.pop('queryType', None)
            
        # 处理信息抽取字段
        if not settings.ENABLE_INFO_EXTRACTION:
            self.__dict__.pop('extractedInfo', None)
            
        return self

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True
        exclude_none = True  # 序列化时排除值为None的字段


# 查询改写响应模型
class QueryRewriteResponse(BaseModel):
    code: str = "0000"
    message: str = "success"
    success: bool = True
    data: Optional[QueryRewriteResults] = None

    class Config:
        exclude_none = True  # 序列化时排除值为None的字段
