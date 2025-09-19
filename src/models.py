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
    historyList: Optional[List[str]] = []
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
        if len(v) > 5:
            return v[-5:]  # 只保留最近的5条历史记录
        return v
    
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
