# Date    : 2024/7/17 15:01
# File    : api_server.py
# Desc    : 查询改写服务API
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com

import os
import time
import yaml
import asyncio
import requests
import json
import re
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import traceback

from models import QueryRewriteRequest, QueryRewriteResponse, QueryRewriteResults, ExtractionRule
from constants import (
    SERVICE_SUCCESS, UNK_ERROR, SERVICE_ERROR, REQ_PARAM_ERROR,
    JUDGE_PROMPTS_FILE, REWRITE_PROMPTS_FILE, SUMMARY_PROMPTS_FILE, CLASSIFY_PROMPTS_FILE,
    EXTRACTION_PROMPTS_FILE, QUERY_REWRITE_ROUTE, API_PREFIX, QueryType
)
from exceptions import CustomException, custom_exception_handler, req_valid_exception_handler
from settings import settings
from logger import logger, formatted_message


# 构建判断模型服务的URL
def get_judge_service_url():
    protocol = settings.protocol if hasattr(settings, 'protocol') else "http"
    host = settings.judge_model_internal_host if hasattr(settings, 'judge_model_internal_host') else "localhost"
    port = settings.judge_model_internal_port if hasattr(settings, 'judge_model_internal_port') else "12401"
    endpoint = settings.judge_model_internal_endpoint if hasattr(settings, 'judge_model_internal_endpoint') else "/yun/ai/internal/judge"
    
    return f"{protocol}://{host}:{port}{endpoint}"


# 构建改写模型服务的URL
def get_rewrite_service_url():
    protocol = settings.protocol if hasattr(settings, 'protocol') else "http"
    host = settings.rewrite_model_internal_host if hasattr(settings, 'rewrite_model_internal_host') else "localhost"
    port = settings.rewrite_model_internal_port if hasattr(settings, 'rewrite_model_internal_port') else "12402"
    endpoint = settings.rewrite_model_internal_endpoint if hasattr(settings, 'rewrite_model_internal_endpoint') else "/yun/ai/internal/rewrite"
    
    return f"{protocol}://{host}:{port}{endpoint}"


# 构建总结判断模型服务的URL
def get_summary_service_url():
    protocol = settings.protocol if hasattr(settings, 'protocol') else "http"
    host = settings.summary_model_internal_host if hasattr(settings, 'summary_model_internal_host') else "localhost"
    port = settings.summary_model_internal_port if hasattr(settings, 'summary_model_internal_port') else "12403"
    endpoint = settings.summary_model_internal_endpoint if hasattr(settings, 'summary_model_internal_endpoint') else "/yun/ai/internal/summary"
    
    return f"{protocol}://{host}:{port}{endpoint}"


# 构建查询分类模型服务的URL
def get_classify_service_url():
    protocol = settings.protocol if hasattr(settings, 'protocol') else "http"
    host = settings.classify_model_internal_host if hasattr(settings, 'classify_model_internal_host') else "localhost"
    port = settings.classify_model_internal_port if hasattr(settings, 'classify_model_internal_port') else "12404"
    endpoint = settings.classify_model_internal_endpoint if hasattr(settings, 'classify_model_internal_endpoint') else "/yun/ai/internal/classify"
    
    return f"{protocol}://{host}:{port}{endpoint}"


# 构建信息抽取模型服务的URL
def get_extraction_service_url():
    protocol = settings.protocol if hasattr(settings, 'protocol') else "http"
    host = settings.extraction_model_internal_host if hasattr(settings, 'extraction_model_internal_host') else "0.0.0.0"
    port = settings.extraction_model_internal_port if hasattr(settings, 'extraction_model_internal_port') else "12319"
    endpoint = settings.extraction_model_internal_endpoint if hasattr(settings, 'extraction_model_internal_endpoint') else "/yun/ai/internal/llm/qwen/3b"
    
    return f"{protocol}://{host}:{port}{endpoint}"


# 大模型服务URL
JUDGE_SERVICE_URL = get_judge_service_url()
REWRITE_SERVICE_URL = get_rewrite_service_url()
SUMMARY_SERVICE_URL = get_summary_service_url()
CLASSIFY_SERVICE_URL = get_classify_service_url()
EXTRACTION_SERVICE_URL = get_extraction_service_url()
formatted_message(logger, "判断模型服务URL", {"url": JUDGE_SERVICE_URL})
formatted_message(logger, "改写模型服务URL", {"url": REWRITE_SERVICE_URL})
formatted_message(logger, "总结判断模型服务URL", {"url": SUMMARY_SERVICE_URL})
formatted_message(logger, "查询分类模型服务URL", {"url": CLASSIFY_SERVICE_URL})
formatted_message(logger, "信息抽取模型服务URL", {"url": EXTRACTION_SERVICE_URL})


# 加载提示词模板
def load_prompts():
    try:
        # 加载判断提示词
        judge_prompts_path = os.path.join(os.path.dirname(__file__), JUDGE_PROMPTS_FILE)
        with open(judge_prompts_path, 'r', encoding='utf-8') as f:
            judge_prompts = yaml.safe_load(f)
        
        # 加载改写提示词
        rewrite_prompts_path = os.path.join(os.path.dirname(__file__), REWRITE_PROMPTS_FILE)
        with open(rewrite_prompts_path, 'r', encoding='utf-8') as f:
            rewrite_prompts = yaml.safe_load(f)
            
        # 加载总结判断提示词
        summary_prompts_path = os.path.join(os.path.dirname(__file__), SUMMARY_PROMPTS_FILE)
        with open(summary_prompts_path, 'r', encoding='utf-8') as f:
            summary_prompts = yaml.safe_load(f)
            
        # 加载查询分类提示词
        classify_prompts_path = os.path.join(os.path.dirname(__file__), CLASSIFY_PROMPTS_FILE)
        with open(classify_prompts_path, 'r', encoding='utf-8') as f:
            classify_prompts = yaml.safe_load(f)
            
        # 加载信息抽取提示词
        extraction_prompts_path = os.path.join(os.path.dirname(__file__), EXTRACTION_PROMPTS_FILE)
        with open(extraction_prompts_path, 'r', encoding='utf-8') as f:
            extraction_prompts = yaml.safe_load(f)
        
        formatted_message(logger, "提示词模板加载成功")
        return {
            "judge": judge_prompts,
            "rewrite": rewrite_prompts,
            "summary": summary_prompts,
            "classify": classify_prompts,
            "extraction": extraction_prompts
        }
    except Exception as e:
        formatted_message(
            logger,
            "异常捕获（加载提示词模板失败）",
            {
                "error": str(e)
            },
            level="error"
        )
        return {"judge": {}, "rewrite": {}, "summary": {}, "classify": {}, "extraction": {}}


# 全局变量存储提示词模板
prompts = load_prompts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    # 启动时执行操作
    yield
    # 关闭时执行清理操作


app = FastAPI(
    title="Query Rewrite API",
    description="基于大模型的查询改写服务",
    version=settings.version,
    docs_url="/docs" if settings.env in ['dev', 'test'] else None,
    redoc_url="/redoc" if settings.env in ['dev', 'test'] else None,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册异常处理器
app.add_exception_handler(CustomException, custom_exception_handler)
app.add_exception_handler(RequestValidationError, req_valid_exception_handler)


def call_llm_for_judge(query: str, history_queries: List[str], request_id: str = ""):
    """调用判断模型服务判断是否需要改写"""
    try:
        global prompts
        
        if not prompts or "judge" not in prompts or not prompts["judge"]:
            formatted_message(
                logger,
                "异常捕获（全局提示词模板变量为空，重新尝试加载）",
                level="error"
            )
            prompts = load_prompts()
        
        # 获取判断模型的提示词模板，使用settings中的版本
        judge_prompts = prompts["judge"]
        prompt_template = judge_prompts.get(settings.JUDGE_PROMPT_VERSION, "")
        
        if not prompt_template:
            formatted_message(
                logger,
                "异常捕获（未找到对应版本的判断提示词）",
                {
                    "version": settings.JUDGE_PROMPT_VERSION
                },
                level="error"
            )
            return False
        
        # 格式化提示词，明确处理空历史查询的情况
        history_text = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(history_queries)]) if history_queries else ""
        prompt = prompt_template.format(
            history_queries=history_text,
            current_query=query
        )
        
        formatted_message(
            logger,
            "判断模型 - 构造的Prompt",
            {
                "requestId": request_id,
                "prompt": prompt
            },
            level="debug"
        )
        
        # 构建请求
        payload = {
            "request_id": request_id,
            "input": prompt,
            "max_new_token": settings.MAX_NEW_TOKEN if hasattr(settings, 'MAX_NEW_TOKEN') else 256,
            "temperature": settings.TEMPERATURE if hasattr(settings, 'TEMPERATURE') else 0.1,
            "top_p": settings.TOP_P if hasattr(settings, 'TOP_P') else 0.9,
            "streaming": False
        }
        
        # 记录请求开始时间
        st_request = time.time()
        
        # 发送请求
        timeout_value = settings.TIMEOUT if hasattr(settings, 'TIMEOUT') else 10
        response = requests.post(JUDGE_SERVICE_URL, json=payload, timeout=timeout_value)
        response.raise_for_status()
        result = response.json()
        
        # # 记录请求耗时
        # request_time = int((time.time() - st_request) * 1000)
        # formatted_message(
        #     logger,
        #     "判断模型请求完成",
        #     {
        #         "request_time": f"{request_time}ms"
        #     }
        # )
        
        # 验证响应是否成功
        if not result.get("success", False):
            error_msg = result.get('message', '未知错误')
            raise Exception(f"判断模型服务返回错误: {error_msg}")
        
        # 获取结果
        raw_output = result.get("data", {}).get("result", "false")
        formatted_message(
            logger,
            "判断模型 - LLM原始输出",
            {
                "requestId": request_id,
                "raw_output": raw_output
            },
            level="debug"
        )
        
        result_text = raw_output.strip().lower()
        formatted_message(
            logger,
            "判断模型 - 后处理结果",
            {
                "requestId": request_id,
                "processed_output": result_text
            },
            level="debug"
        )
        
        # 解析结果
        if "true" in result_text:
            return True
        return False
    
    except Exception as e:
        formatted_message(
            logger,
            "异常捕获（调用判断模型服务失败）",
            {
                "error": str(e)
            },
            level="error"
        )
        return False


def call_llm_for_rewrite(query: str, history_queries: List[str], request_id: str = ""):
    """调用改写模型服务进行查询改写"""
    try:
        global prompts
        
        if not prompts or "rewrite" not in prompts or not prompts["rewrite"]:
            formatted_message(
                logger,
                "异常捕获（全局提示词模板变量为空，重新尝试加载）",
                level="error"
            )
            prompts = load_prompts()
        
        # 获取改写模型的提示词模板，使用settings中的版本
        rewrite_prompts = prompts["rewrite"]
        prompt_template = rewrite_prompts.get(settings.REWRITE_PROMPT_VERSION, "")
        
        if not prompt_template:
            formatted_message(
                logger,
                "异常捕获（未找到对应版本的改写提示词）",
                {
                    "version": settings.REWRITE_PROMPT_VERSION
                },
                level="error"
            )
            return query
        
        # 格式化提示词
        history_text = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(history_queries)]) if history_queries else ""
        
        prompt = prompt_template.format(
            history_text=history_text,
            current_query=query
        )
        
        formatted_message(
            logger,
            "改写模型 - 构造的Prompt",
            {
                "requestId": request_id,
                "prompt": prompt
            },
            level="debug"
        )
        
        # 构建请求
        payload = {
            "request_id": request_id,
            "input": prompt,
            "max_new_token": settings.MAX_NEW_TOKEN if hasattr(settings, 'MAX_NEW_TOKEN') else 256,
            "temperature": settings.TEMPERATURE if hasattr(settings, 'TEMPERATURE') else 0.1,
            "top_p": settings.TOP_P if hasattr(settings, 'TOP_P') else 0.9,
            "streaming": False
        }
        
        # 记录请求开始时间
        st_request = time.time()
        
        # 发送请求
        timeout_value = settings.TIMEOUT if hasattr(settings, 'TIMEOUT') else 10
        response = requests.post(REWRITE_SERVICE_URL, json=payload, timeout=timeout_value)
        response.raise_for_status()
        result = response.json()
        
        # # 记录请求耗时
        # request_time = int((time.time() - st_request) * 1000)
        # formatted_message(
        #     logger,
        #     "改写模型请求完成",
        #     {
        #         "request_time": f"{request_time}ms"
        #     }
        # )
        
        # 验证响应是否成功
        if not result.get("success", False):
            error_msg = result.get('message', '未知错误')
            raise Exception(f"改写模型服务返回错误: {error_msg}")
        
        # 获取结果
        raw_output = result.get("data", {}).get("result", "")
        formatted_message(
            logger,
            "改写模型 - LLM原始输出",
            {
                "requestId": request_id,
                "raw_output": raw_output
            },
            level="debug"
        )
        
        result_text = raw_output.strip()
        formatted_message(
            logger,
            "改写模型 - 后处理结果 (strip)",
            {
                "requestId": request_id,
                "processed_output": result_text
            },
            level="debug"
        )
        
        # 如果返回为空，使用原始查询
        if not result_text or result_text.isspace():
            return query
            
        return result_text
    
    except Exception as e:
        formatted_message(
            logger,
            "异常捕获（调用改写模型服务失败）",
            {
                "error": str(e)
            },
            level="error"
        )
        return query


def call_judge_and_rewrite_pipeline(query: str, history_queries: List[str], request_id: str = ""):
    """判断并改写查询的pipeline，将判断和改写合并为一个流水线处理"""
    try:
        # 第一步：判断是否需要改写
        need_rewrite = call_llm_for_judge(query, history_queries, request_id)
        
        # 第二步：如果需要改写，则进行改写；否则返回原始查询
        if need_rewrite:
            rewritten_query = call_llm_for_rewrite(query, history_queries, request_id)
            # 清理输出格式
            rewritten_query = clean_output(rewritten_query)
            return {
                "needRewrite": True,
                "result": rewritten_query
            }
        else:
            return {
                "needRewrite": False,
                "result": query
            }
    
    except Exception as e:
        formatted_message(
            logger,
            "异常捕获（判断改写pipeline失败）",
            {
                "error": str(e)
            },
            level="error"
        )
        return {
            "needRewrite": False,
            "result": query
        }


def call_llm_for_summary_check(query: str, request_id: str = ""):
    """调用总结判断模型服务判断是否为总结类查询并提取关键词"""
    try:
        global prompts

        # 获取总结判断模型的提示词模板，使用settings中的版本
        summary_prompts = prompts["summary"]
        prompt_template = summary_prompts.get(settings.SUMMARY_PROMPT_VERSION, "")

        # 格式化提示词
        prompt = prompt_template.format(
            current_query=query
        )

        formatted_message(
            logger,
            "总结判断模型 - 构造的Prompt",
            {
                "requestId": request_id,
                "prompt": prompt
            },
            level="debug"
        )

        # 构建请求
        payload = {
            "request_id": request_id,
            "input": prompt,
            "max_new_token": settings.MAX_NEW_TOKEN if hasattr(settings, 'MAX_NEW_TOKEN') else 256,
            "temperature": settings.TEMPERATURE if hasattr(settings, 'TEMPERATURE') else 0.1,
            "top_p": settings.TOP_P if hasattr(settings, 'TOP_P') else 0.9,
            "streaming": False
        }
        
        # 发送请求
        timeout_value = settings.TIMEOUT if hasattr(settings, 'TIMEOUT') else 10
        response = requests.post(SUMMARY_SERVICE_URL, json=payload, timeout=timeout_value)
        response.raise_for_status()
        result = response.json()
        
        # 验证响应是否成功
        if not result.get("success", False):
            error_msg = result.get('message', '未知错误')
            raise Exception(f"总结判断模型服务返回错误: {error_msg}")
        
        # 获取结果，检查data字段
        if "data" not in result:
            return {"isSummary": False, "keywords": []}
            
        data = result["data"]
        if not isinstance(data, dict):
            return {"isSummary": False, "keywords": []}
            
        # 获取result字段
        if "result" not in data:
            return {"isSummary": False, "keywords": []}
            
        result_text = data["result"]
        if not isinstance(result_text, str):
            result_text = str(result_text)
        
        result_text = result_text.strip()
        
        formatted_message(
            logger,
            "总结判断模型 - LLM原始输出",
            {
                "requestId": request_id,
                "raw_output": result_text
            },
            level="debug"
        )
        
        # 解析JSON结果
        try:
            # 清理结果文本，处理可能的非标准JSON格式
            cleaned_text = result_text.strip()
            
            # 如果返回的是空文本，直接返回默认值
            if not cleaned_text:
                return {"isSummary": False, "keywords": []}
            
            # 查找JSON开始和结束的位置
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                formatted_message(
                    logger,
                    "总结判断模型 - 提取的JSON字符串",
                    {
                        "requestId": request_id,
                        "json_str": json_str
                    },
                    level="debug"
                )
                
                # 替换可能导致JSON解析失败的字符
                json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                json_str = json_str.replace('true/false', 'false')  # 处理可能的"true/false"文本
                json_str = json_str.replace('True', 'true').replace('False', 'false')
                json_str = ' '.join(json_str.split())  # 规范化空白字符
                
                try:
                    summary_result = json.loads(json_str)
                    formatted_message(
                        logger,
                        "总结判断模型 - JSON解析结果",
                        {
                            "requestId": request_id,
                            "parsed_json": summary_result
                        },
                        level="debug"
                    )
                except json.JSONDecodeError as je:
                    # 如果是简单的布尔值问题，尝试直接构造结果
                    if '"isSummary"' in json_str:
                        is_summary = 'true' in json_str.lower()
                        return {"isSummary": is_summary, "keywords": []}
                    return {"isSummary": False, "keywords": []}
            else:
                # 如果找不到有效的JSON格式，尝试直接从文本判断
                is_summary = 'true' in cleaned_text.lower() and 'summary' in cleaned_text.lower()
                return {"isSummary": is_summary, "keywords": []}
            
            # 确保结果包含必要的字段
            is_summary = summary_result.get("isSummary", False)
            # 将字符串"true"或"false"转换为布尔值
            if isinstance(is_summary, str):
                is_summary = is_summary.lower() == "true"
                
            keywords = summary_result.get("keywords", [])
            
            # 确保关键词是列表类型
            if not isinstance(keywords, list):
                if isinstance(keywords, str):
                    # 如果是逗号分隔的字符串，转换为列表
                    keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                else:
                    keywords = []
            
            return {
                "isSummary": is_summary, 
                "keywords": keywords
            }
            
        except Exception as e:
            formatted_message(
                logger,
                "异常捕获（解析总结判断模型结果失败）",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "result_text": result_text
                },
                level="error"
            )
            
            # 尝试简单解析，如果结果包含"true"，则判断为总结类
            is_summary = "true" in result_text.lower() and "summary" in result_text.lower()
            keywords = []
            
            return {"isSummary": is_summary, "keywords": keywords}
    
    except Exception as e:
        formatted_message(
            logger,
            "异常捕获（调用总结判断模型服务失败）",
            {
                "error": str(e),
                "error_type": type(e).__name__
            },
            level="error"
        )
        return {"isSummary": False, "keywords": []}


def call_llm_for_classify(query: str, request_id: str = ""):
    """调用查询分类模型服务判断查询类型"""
    try:
        global prompts

        # 获取查询分类模型的提示词模板，使用settings中的版本
        classify_prompts = prompts["classify"]
        prompt_template = classify_prompts.get(settings.CLASSIFY_PROMPT_VERSION, "")
        
        if not prompt_template:
            formatted_message(
                logger,
                "异常捕获（未找到对应版本的查询分类提示词）",
                {
                    "version": settings.CLASSIFY_PROMPT_VERSION
                },
                level="error"
            )
            return QueryType.OTHER

        # 格式化提示词
        prompt = prompt_template.format(
            current_query=query
        )

        formatted_message(
            logger,
            "查询分类模型 - 构造的Prompt",
            {
                "requestId": request_id,
                "prompt": prompt
            },
            level="debug"
        )

        # 构建请求
        payload = {
            "request_id": request_id,
            "input": prompt,
            "max_new_token": settings.MAX_NEW_TOKEN if hasattr(settings, 'MAX_NEW_TOKEN') else 256,
            "temperature": settings.TEMPERATURE if hasattr(settings, 'TEMPERATURE') else 0.1,
            "top_p": settings.TOP_P if hasattr(settings, 'TOP_P') else 0.9,
            "streaming": False
        }
        
        # 发送请求
        timeout_value = settings.TIMEOUT if hasattr(settings, 'TIMEOUT') else 10
        response = requests.post(CLASSIFY_SERVICE_URL, json=payload, timeout=timeout_value)
        response.raise_for_status()
        result = response.json()
        
        # 验证响应是否成功
        if not result.get("success", False):
            error_msg = result.get('message', '未知错误')
            raise Exception(f"查询分类模型服务返回错误: {error_msg}")
        
        # 获取结果，检查data字段
        if "data" not in result:
            return QueryType.OTHER
            
        data = result["data"]
        if not isinstance(data, dict):
            return QueryType.OTHER
            
        # 获取result字段
        if "result" not in data:
            return QueryType.OTHER
            
        result_text = data["result"]
        if not isinstance(result_text, str):
            result_text = str(result_text)
        
        result_text = result_text.strip()
        
        formatted_message(
            logger,
            "查询分类模型 - LLM原始输出",
            {
                "requestId": request_id,
                "raw_output": result_text
            },
            level="debug"
        )
        
        # 解析JSON结果
        try:
            # 清理结果文本，处理可能的非标准JSON格式
            cleaned_text = result_text.strip()
            
            # 如果返回的是空文本，直接返回默认值
            if not cleaned_text:
                return QueryType.OTHER
            
            # 查找JSON开始和结束的位置
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                formatted_message(
                    logger,
                    "查询分类模型 - 提取的JSON字符串",
                    {
                        "requestId": request_id,
                        "json_str": json_str
                    },
                    level="debug"
                )
                
                # 替换可能导致JSON解析失败的字符
                json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                json_str = ' '.join(json_str.split())  # 规范化空白字符
                
                try:
                    classify_result = json.loads(json_str)
                    formatted_message(
                        logger,
                        "查询分类模型 - JSON解析结果",
                        {
                            "requestId": request_id,
                            "parsed_json": classify_result
                        },
                        level="debug"
                    )
                except json.JSONDecodeError as je:
                    formatted_message(
                        logger,
                        "异常捕获（解析查询分类JSON失败）",
                        {
                            "error": str(je),
                            "json_str": json_str
                        },
                        level="error"
                    )
                    return QueryType.OTHER
            else:
                # 直接从文本尝试提取数字
                try:
                    # 寻找数字
                    import re
                    numbers = re.findall(r'\d+', cleaned_text)
                    if numbers:
                        query_type = int(numbers[0])
                        if 0 <= query_type <= 3:  # 确保值在有效范围内
                            return query_type
                    return QueryType.OTHER
                except:
                    return QueryType.OTHER
            
            # 确保结果包含必要的字段
            query_type = classify_result.get("queryType", QueryType.OTHER)
            
            # 确保类型是整数
            try:
                query_type = int(query_type)
                # 确保值在枚举范围内
                if query_type not in [member.value for member in QueryType]:
                    query_type = QueryType.OTHER
            except:
                query_type = QueryType.OTHER
                
            return query_type
            
        except Exception as e:
            formatted_message(
                logger,
                "异常捕获（解析查询分类模型结果失败）",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "result_text": result_text
                },
                level="error"
            )
            
            return QueryType.OTHER
    
    except Exception as e:
        formatted_message(
            logger,
            "异常捕获（调用查询分类模型服务失败）",
            {
                "error": str(e),
                "error_type": type(e).__name__
            },
            level="error"
        )
        return QueryType.OTHER


def call_llm_for_extraction(query: str, extraction_rule: ExtractionRule, request_id: str = ""):
    """调用信息抽取模型服务提取查询中的特定信息"""
    try:
        global prompts

        # 获取信息抽取模型的提示词模板
        extraction_prompts = prompts["extraction"]
        prompt_template = extraction_prompts.get(settings.EXTRACTION_PROMPT_VERSION, "")
        
        if not prompt_template:
            formatted_message(
                logger,
                "异常捕获（未找到对应版本的信息抽取提示词）",
                {
                    "version": settings.EXTRACTION_PROMPT_VERSION
                },
                level="error"
            )
            return []

        # 详细记录extraction_rule的内容
        # formatted_message(
        #     logger,
        #     "信息抽取规则",
        #     {
        #         "desc": extraction_rule.desc,
        #         "valueType": extraction_rule.valueType,
        #         "keyName": extraction_rule.keyName,
        #         "defaultValue": extraction_rule.defaultValue
        #     }
        # )

        # 格式化提示词
        default_value = "null" if extraction_rule.defaultValue is None else extraction_rule.defaultValue
        key_name_value = extraction_rule.keyName
        
        # 使用安全的方式格式化提示词
        prompt = prompt_template.format(
            extraction_desc=extraction_rule.desc,
            value_type=extraction_rule.valueType,
            key_name=key_name_value,
            default_value=default_value,
            query=query
        )
        
        # 记录完整的提示词
        formatted_message(
            logger,
            "信息抽取提示词",
            {
                "requestId": request_id,
                "prompt": prompt
            },
            level="debug"
        )

        # 构建请求
        payload = {
            "request_id": request_id,
            "input": prompt,
            "max_new_token": settings.MAX_NEW_TOKEN if hasattr(settings, 'MAX_NEW_TOKEN') else 256,
            "temperature": settings.TEMPERATURE if hasattr(settings, 'TEMPERATURE') else 0.1,
            "top_p": settings.TOP_P if hasattr(settings, 'TOP_P') else 0.9,
            "streaming": False
        }
        
        # 记录请求体
        formatted_message(
            logger,
            "信息抽取请求体",
            {
                "requestId": request_id,
                "url": EXTRACTION_SERVICE_URL,
                "payload": payload
            },
            level="debug"
        )
        
        # 发送请求
        timeout_value = settings.TIMEOUT if hasattr(settings, 'TIMEOUT') else 10
        response = requests.post(EXTRACTION_SERVICE_URL, json=payload, timeout=timeout_value)
        response.raise_for_status()
        result = response.json()
        
        # 记录响应
        formatted_message(
            logger,
            "信息抽取模型响应",
            {
                "requestId": request_id,
                "response": result
            },
            level="debug"
        )
        
        # 验证响应是否成功
        if not result.get("success", False):
            error_msg = result.get('message', '未知错误')
            raise Exception(f"信息抽取模型服务返回错误: {error_msg}")
        
        # 获取结果
        if "data" not in result:
            formatted_message(
                logger,
                "信息抽取响应中缺少data字段",
                level="error"
            )
            return [{key_name_value: extraction_rule.defaultValue}]
            
        data = result["data"]
        if not isinstance(data, dict):
            formatted_message(
                logger,
                "信息抽取响应中data字段不是字典类型",
                {
                    "data_type": type(data).__name__,
                    "data": data
                },
                level="error"
            )
            return [{key_name_value: extraction_rule.defaultValue}]
            
        # 获取result字段
        if "result" not in data:
            formatted_message(
                logger,
                "信息抽取响应中缺少result字段",
                {
                    "data": data
                },
                level="error"
            )
            return [{key_name_value: extraction_rule.defaultValue}]
            
        result_text = data["result"]
        if not isinstance(result_text, str):
            result_text = str(result_text)
        
        result_text = result_text.strip()
        
        # 记录原始结果文本
        formatted_message(
            logger,
            "信息抽取结果文本",
            {
                "requestId": request_id,
                "result_text": result_text
            },
            level="debug"
        )
        
        # 解析JSON结果
        try:
            # 清理结果文本，处理可能的非标准JSON格式
            cleaned_text = result_text.strip()
            
            # 如果返回的是空文本，直接返回默认值
            if not cleaned_text:
                formatted_message(
                    logger,
                    "信息抽取结果为空",
                    level="warning"
                )
                return [{key_name_value: extraction_rule.defaultValue}]
            
            # 查找JSON开始和结束的位置
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            formatted_message(
                logger,
                "JSON解析范围",
                {
                    "requestId": request_id,
                    "json_start": json_start,
                    "json_end": json_end,
                    "json_text": cleaned_text[json_start:json_end] if json_start >= 0 and json_end > json_start else "未找到JSON"
                },
                level="debug"
            )
            
            # 如果没有找到完整的JSON结构，尝试通过正则表达式直接提取
            if json_start < 0 or json_end <= json_start:
                formatted_message(
                    logger,
                    "未找到有效的JSON结构，尝试使用正则表达式提取",
                    level="warning"
                )
                
                # 针对不同的值类型使用不同的提取方法
                if extraction_rule.valueType.lower() == "int":
                    import re
                    numbers = re.findall(r'\d+', cleaned_text)
                    if numbers:
                        value = int(numbers[0])
                        formatted_message(
                            logger,
                            "直接提取数字成功",
                            {
                                "value": value
                            }
                        )
                        return [{key_name_value: value}]
                elif extraction_rule.valueType.lower() == "string":
                    # 尝试提取引号中的内容，这可能是字符串值
                    import re
                    strings = re.findall(r'"([^"]*)"', cleaned_text)
                    if strings and len(strings) >= 2:  # 假设至少有一对键值
                        value = strings[1]  # 第二个匹配项可能是值
                        formatted_message(
                            logger,
                            "直接提取字符串成功",
                            {
                                "value": value
                            }
                        )
                        return [{key_name_value: value}]
                    
                # 如果提取失败，返回默认值
                return [{key_name_value: extraction_rule.defaultValue}]
            
            # 提取JSON文本并进行清理
            json_str = cleaned_text[json_start:json_end]
            
            # 替换可能导致JSON解析失败的字符
            json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            json_str = ' '.join(json_str.split())  # 规范化空白字符
            
            # 特殊处理: 检查和修复常见的格式问题
            # 1. 检查是否包含 "{key_name}" 这样的模式
            if '{' in json_str and '}' in json_str and not json_str.startswith('{{'):
                # 可能是模板没有被正确替换
                pattern = r'"{([^}]+)}"'
                import re
                matches = re.findall(pattern, json_str)
                for match in matches:
                    json_str = json_str.replace(f'"{{{match}}}"', f'"{match}"')
            
            formatted_message(
                logger,
                "待解析的JSON字符串",
                {
                    "json_str": json_str
                }
            )
            
            # 尝试解析JSON
            try:
                extraction_result = json.loads(json_str)
                formatted_message(
                    logger,
                    "JSON解析结果",
                    {
                        "requestId": request_id,
                        "extraction_result": extraction_result
                    },
                    level="debug"
                )
                
                # 确保抽取结果中包含指定的键
                if key_name_value in extraction_result:
                    return [{key_name_value: extraction_result[key_name_value]}]
                else:
                    # 如果LLM返回的JSON中键名不符合预期，尝试修复
                    # 例如：LLM可能返回了 {"requireWordNums": 15000} 而不是指定的键名
                    fixed_result = {}
                    
                    if len(extraction_result) == 1:
                        # 如果只有一个键值对，假设值是正确的
                        value = list(extraction_result.values())[0]
                        fixed_result = {key_name_value: value}
                        formatted_message(
                            logger,
                            "修复键名",
                            {
                                "original_keys": list(extraction_result.keys()),
                                "fixed_result": fixed_result
                            }
                        )
                        return [fixed_result]
                    
                    # 否则使用默认值
                    formatted_message(
                        logger,
                        "使用默认值",
                        {
                            "key": key_name_value,
                            "defaultValue": extraction_rule.defaultValue
                        }
                    )
                    return [{key_name_value: extraction_rule.defaultValue}]
                    
            except json.JSONDecodeError as je:
                formatted_message(
                    logger,
                    "异常捕获（解析信息抽取JSON失败）",
                    {
                        "error": str(je),
                        "error_position": je.pos,
                        "json_str": json_str
                    },
                    level="error"
                )
                
                # 尝试解析字符串中的特定格式
                import re
                
                if extraction_rule.valueType.lower() == "int":
                    # 尝试找到类似 "requireWordNums": 15000 的模式
                    pattern = r'"' + re.escape(key_name_value) + r'"\s*:\s*(\d+)'
                    matches = re.findall(pattern, json_str)
                    if matches:
                        value = int(matches[0])
                        formatted_message(
                            logger,
                            "通过正则表达式提取键值对成功",
                            {
                                "key": key_name_value,
                                "value": value
                            }
                        )
                        return [{key_name_value: value}]
                    
                    # 尝试找到任何数字
                    numbers = re.findall(r'\d+', json_str)
                    if numbers:
                        value = int(numbers[0])
                        formatted_message(
                            logger,
                            "通过正则表达式提取数字成功",
                            {
                                "value": value
                            }
                        )
                        return [{key_name_value: value}]
                elif extraction_rule.valueType.lower() == "string":
                    # 尝试找到类似 "reportTopic": "人工智能" 的模式
                    pattern = r'"' + re.escape(key_name_value) + r'"\s*:\s*"([^"]+)"'
                    matches = re.findall(pattern, json_str)
                    if matches:
                        value = matches[0]
                        formatted_message(
                            logger,
                            "通过正则表达式提取键值对成功",
                            {
                                "key": key_name_value,
                                "value": value
                            }
                        )
                        return [{key_name_value: value}]
                    
                    # 尝试找到任何引号中的内容
                    strings = re.findall(r'"([^"]+)"', json_str)
                    if len(strings) >= 2:  # 至少有一对键值
                        for i in range(len(strings)):
                            if strings[i] == key_name_value and i + 1 < len(strings):
                                return [{key_name_value: strings[i+1]}]
                        
                        # 如果没找到键名匹配，假设第二个字符串是值
                        if len(strings) >= 2:
                            formatted_message(
                                logger,
                                "通过正则表达式提取字符串成功",
                                {
                                    "value": strings[1]
                                }
                            )
                            return [{key_name_value: strings[1]}]
                
                # 所有尝试都失败，返回默认值
                return [{key_name_value: extraction_rule.defaultValue}]
            
        except Exception as e:
            formatted_message(
                logger,
                "异常捕获（解析信息抽取模型结果失败）",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "result_text": result_text
                },
                level="error"
            )
            # 返回默认值
            return [{key_name_value: extraction_rule.defaultValue}]
    
    except Exception as e:
        formatted_message(
            logger,
            "异常捕获（调用信息抽取模型服务失败）",
            {
                "error": str(e),
                "error_type": type(e).__name__
            },
            level="error"
        )
        # 返回默认值
        if extraction_rule and hasattr(extraction_rule, 'keyName') and hasattr(extraction_rule, 'defaultValue'):
            return [{extraction_rule.keyName: extraction_rule.defaultValue}]
        return []


def call_llm_for_batch_extraction(query: str, extraction_rules: List[ExtractionRule], request_id: str = ""):
    """调用信息抽取模型服务一次性提取查询中的多个特定信息"""
    try:
        global prompts

        # 获取信息抽取模型的提示词模板
        extraction_prompts = prompts["extraction"]
        prompt_template = extraction_prompts.get(settings.EXTRACTION_PROMPT_VERSION, "")
        
        if not prompt_template:
            formatted_message(
                logger,
                "异常捕获（未找到对应版本的信息抽取提示词模板）",
                {
                    "version": settings.EXTRACTION_PROMPT_VERSION
                },
                level="error"
            )
            return create_default_results(extraction_rules)

        # 将抽取规则转换为JSON格式
        extraction_rules_json = json.dumps([{
            "desc": rule.desc,
            "valueType": rule.valueType,
            "keyName": rule.keyName,
            "defaultValue": "null" if rule.defaultValue is None else rule.defaultValue
        } for rule in extraction_rules], ensure_ascii=False, indent=2)

        # 格式化提示词 - 修复方法：使用带有转义的格式化
        # 1. 将所有非格式化占位符的花括号变成双花括号以转义
        # 2. 确保格式化正确应用到实际的占位符
        try:
            # 在模板中用双花括号转义所有花括号
            escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
            # 再将实际的占位符恢复为单花括号
            escaped_template = escaped_template.replace("{{extraction_rules_json}}", "{extraction_rules_json}")
            escaped_template = escaped_template.replace("{{query}}", "{query}")
            
            # 使用转义后的模板进行格式化
            prompt = escaped_template.format(
                extraction_rules_json=extraction_rules_json,
                query=query
            )
        except KeyError as ke:
            # 如果仍然出现KeyError，记录详细错误并尝试备选方案
            formatted_message(
                logger,
                "提示词模板转义失败",
                {
                    "error": str(ke),
                    "template_length": len(prompt_template)
                },
                level="error"
            )
            
            # 备选方案：手动替换占位符
            prompt = prompt_template
            prompt = prompt.replace("{extraction_rules_json}", extraction_rules_json)
            prompt = prompt.replace("{query}", query)
            
            formatted_message(
                logger,
                "使用备选方案替换占位符",
                {
                    "success": True
                }
            )
        
        # # 记录完整的提示词内容（用于debug）
        formatted_message(
            logger,
            "批量信息抽取 - 构造的Prompt",
            {
                "requestId": request_id,
                "prompt": prompt
            },
            level="debug"
        )

        # 构建请求
        payload = {
            "request_id": request_id,
            "input": prompt,
            "max_new_token": settings.MAX_NEW_TOKEN if hasattr(settings, 'MAX_NEW_TOKEN') else 256,
            "temperature": settings.TEMPERATURE if hasattr(settings, 'TEMPERATURE') else 0.1,
            "top_p": settings.TOP_P if hasattr(settings, 'TOP_P') else 0.9,
            "streaming": False
        }
        
        formatted_message(
            logger,
            "批量信息抽取 - 请求体",
            {
                "requestId": request_id,
                "url": EXTRACTION_SERVICE_URL,
                "payload": payload
            },
            level="debug"
        )
        
        # 发送请求
        timeout_value = settings.TIMEOUT if hasattr(settings, 'TIMEOUT') else 10
        
        # 记录发送请求的时间
        req_start_time = time.time()
        
        try:
            response = requests.post(EXTRACTION_SERVICE_URL, json=payload, timeout=timeout_value)
            
            # 记录请求耗时
            req_duration = time.time() - req_start_time
            
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            formatted_message(
                logger,
                "信息抽取请求异常",
                {
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                level="error"
            )
            raise
        
        formatted_message(
            logger,
            "批量信息抽取 - LLM响应",
            {
                "requestId": request_id,
                "response": result
            },
            level="debug"
        )
        
        # 验证响应是否成功
        if not result.get("success", False):
            error_msg = result.get('message', '未知错误')
            formatted_message(
                logger,
                "信息抽取模型响应失败",
                {
                    "error_message": error_msg
                },
                level="error"
            )
            raise Exception(f"信息抽取模型服务返回错误: {error_msg}")
        
        # 获取结果
        if "data" not in result:
            formatted_message(
                logger,
                "信息抽取响应中缺少data字段",
                level="error"
            )
            return create_default_results(extraction_rules)
            
        data = result["data"]
        
        if not isinstance(data, dict):
            formatted_message(
                logger,
                "信息抽取响应中data字段不是字典类型",
                {
                    "data_type": type(data).__name__,
                    "data": data
                },
                level="error"
            )
            return create_default_results(extraction_rules)
            
        # 获取result字段
        if "result" not in data:
            formatted_message(
                logger,
                "信息抽取响应中缺少result字段",
                {
                    "data": data
                },
                level="error"
            )
            return create_default_results(extraction_rules)
            
        result_text = data["result"]
        if not isinstance(result_text, str):
            result_text = str(result_text)
        
        result_text = result_text.strip()
        
        formatted_message(
            logger,
            "批量信息抽取 - LLM原始输出",
            {
                "requestId": request_id,
                "raw_output": result_text
            },
            level="debug"
        )
        
        # 解析JSON结果
        try:
            # 清理结果文本，处理可能的非标准JSON格式
            cleaned_text = result_text.strip()
            
            # 如果返回的是空文本，直接返回默认值
            if not cleaned_text:
                return create_default_results(extraction_rules)
            
            # 尝试不同的方法查找JSON
            # 1. 尝试直接解析整个文本
            try:
                extraction_result = json.loads(cleaned_text)
                
                formatted_message(
                    logger,
                    "批量信息抽取 - 直接JSON解析成功",
                    {
                        "requestId": request_id,
                        "parsed_json": extraction_result
                    },
                    level="debug"
                )
                
                # 将解析结果转换为所需的格式
                results = []
                # 检查每个抽取规则是否在结果中
                for rule in extraction_rules:
                    if rule.keyName in extraction_result:
                        # 找到对应的键，添加到结果列表
                        results.append({rule.keyName: extraction_result[rule.keyName]})
                    else:
                        # 未找到对应的键，使用默认值
                        results.append({rule.keyName: rule.defaultValue})
                return results
            except json.JSONDecodeError as je:
                # 整个文本不是有效的JSON，继续尝试其他方法
                formatted_message(
                    logger,
                    "整个文本非有效JSON，尝试查找JSON部分",
                    {
                        "requestId": request_id,
                        "error": str(je),
                        "error_position": je.pos,
                        "text_around_error": cleaned_text[max(0, je.pos-20):min(len(cleaned_text), je.pos+20)]
                    },
                    level="debug"
                )
            
            # 2. 查找第一个 { 和最后一个 }
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            # 如果没有找到完整的JSON结构，尝试寻找关键信息
            if json_start < 0 or json_end <= json_start:                
                # 返回默认值
                return extract_fallback(cleaned_text, extraction_rules)
            
            # 提取JSON文本并进行清理
            json_str = cleaned_text[json_start:json_end]
            
            # 替换可能导致JSON解析失败的字符
            original_json_str = json_str
            json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            json_str = ' '.join(json_str.split())  # 规范化空白字符
            
            # 检查和修复明显的JSON格式问题
            # 1. 去除前导或尾随的逗号
            json_str = re.sub(r',\s*}', '}', json_str)
            # 2. 修复常见的引号问题
            json_str = json_str.replace('""', '"')
            
            # 尝试解析JSON
            try:
                # 先尝试使用常规json.loads
                extraction_result = json.loads(json_str)
                
                formatted_message(
                    logger,
                    "批量信息抽取 - 子字符串JSON解析成功",
                    {
                        "requestId": request_id,
                        "parsed_json": extraction_result
                    },
                    level="debug"
                )
                
                # 将解析结果转换为所需的格式
                results = []
                # 检查每个抽取规则是否在结果中
                for rule in extraction_rules:
                    key_name = rule.keyName
                    if key_name in extraction_result:
                        # 找到对应的键，添加到结果列表
                        value = extraction_result[key_name]
                        results.append({key_name: value})
                    else:
                        results.append({key_name: rule.defaultValue})
                
                return results
                    
            except json.JSONDecodeError as je:
                formatted_message(
                    logger,
                    "子字符串JSON解析失败，进入回退逻辑",
                    {
                        "requestId": request_id,
                        "error": str(je),
                        "error_position": je.pos,
                        "json_str": json_str,
                        "json_str_part": json_str[max(0, je.pos-20):min(len(json_str), je.pos+20)],  # 显示错误位置附近的内容
                        "char_at_error": repr(json_str[je.pos]) if je.pos < len(json_str) else None,
                        "char_code_at_error": ord(json_str[je.pos]) if je.pos < len(json_str) else None
                    },
                    level="debug"
                )
                
                return extract_fallback(cleaned_text, extraction_rules)
            
        except Exception as e:
            formatted_message(
                logger,
                "异常捕获（解析信息抽取模型结果失败）",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "result_text": result_text
                },
                level="error"
            )
            return create_default_results(extraction_rules)
    
    except Exception as e:
        formatted_message(
            logger,
            "异常捕获（调用信息抽取模型服务失败）",
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            },
            level="error"
        )
        return create_default_results(extraction_rules)


def extract_fallback(text: str, extraction_rules: List[ExtractionRule]) -> List[Dict[str, Any]]:
    """尝试使用正则表达式等方法从文本中提取信息的备选方案"""
    import re
    
    formatted_message(
        logger,
        "使用备选方案提取信息",
        {
            "text_length": len(text)
        }
    )
    
    results = []
    
    for rule in extraction_rules:
        key_name = rule.keyName
        value_type = rule.valueType.lower()
        default_value = rule.defaultValue
        
        # 记录当前处理的规则
        formatted_message(
            logger,
            f"尝试提取: {key_name}",
            {
                "value_type": value_type,
                "default_value": default_value
            }
        )
        
        extracted_value = None
        
        # 尝试查找键值对模式
        key_pattern = fr'"{key_name}"\s*:\s*'
        key_match = re.search(key_pattern, text)
        
        if key_match:
            start_pos = key_match.end()
            
            if value_type == "int":
                # 数字类型
                num_match = re.search(r'\d+', text[start_pos:start_pos+20])
                if num_match:
                    extracted_value = int(num_match.group(0))
            elif value_type == "string":
                # 字符串类型
                str_match = re.search(r'"([^"]*)"', text[start_pos:start_pos+100])
                if str_match:
                    extracted_value = str_match.group(1)
            elif value_type == "boolean":
                # 布尔类型
                bool_match = re.search(r'true|false', text[start_pos:start_pos+10], re.IGNORECASE)
                if bool_match:
                    extracted_value = bool_match.group(0).lower() == 'true'
        
        # 如果没有找到键值对模式，尝试直接搜索值
        if extracted_value is None:
            if value_type == "int":
                # 查找数字
                num_matches = re.findall(r'\d+', text)
                if num_matches:
                    # 使用第一个匹配的数字
                    extracted_value = int(num_matches[0])
            elif value_type == "string" and key_name in text:
                # 查找与key_name相关的内容
                surrounding_text = text[max(0, text.find(key_name)-50):min(len(text), text.find(key_name)+100)]
                str_matches = re.findall(r'"([^"]*)"', surrounding_text)
                if str_matches and len(str_matches) > 1:
                    # 使用key_name后的第一个字符串
                    extracted_value = str_matches[1]  # 假设第一个是键名，第二个是值
        
        # 记录提取结果
        if extracted_value is not None:
            formatted_message(
                logger,
                f"成功提取: {key_name}",
                {
                    "value": extracted_value
                }
            )
            results.append({key_name: extracted_value})
        else:
            formatted_message(
                logger,
                f"无法提取: {key_name}，使用默认值",
                {
                    "default_value": default_value
                }
            )
            results.append({key_name: default_value})
    
    return results


def create_default_results(extraction_rules: List[ExtractionRule]) -> List[Dict[str, Any]]:
    """为所有抽取规则创建默认结果"""
    return [{rule.keyName: rule.defaultValue} for rule in extraction_rules]


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """中间件: 记录请求处理时间和设置请求ID"""
    # 获取请求ID
    if "requestId" in request.query_params:
        request.state.request_id = request.query_params["requestId"]
    elif request.method == "POST" and request.headers.get("content-type") == "application/json":
        try:
            body = await request.json()
            if "requestId" in body:
                request.state.request_id = body["requestId"]
        except:
            request.state.request_id = "unknown"
    else:
        request.state.request_id = "unknown"
    
    # 记录请求开始时间
    start_time = time.time()
    response = await call_next(request)
    # 计算处理时间
    process_time = int((time.time() - start_time) * 1000)  # 毫秒
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.post(f"{API_PREFIX}{QUERY_REWRITE_ROUTE}", response_model=QueryRewriteResponse)
async def rewrite_query(request: QueryRewriteRequest):
    """
    查询改写API
    """
    start_time = time.time()
    
    # 请求进入日志
    formatted_message(
        logger,
        f"接收请求",
        {
            "requestId": request.requestId,
            "query": request.query,
            "historyList": request.historyList if request.historyList else [],
            "enableExtraction": request.enableExtraction,
            "extractionRules": [rule.dict() for rule in request.extractionRules] if request.extractionRules else None
        }
    )
    
    try:
        # 检查query长度
        if len(request.query) > 128:
            result = QueryRewriteResponse(
                code=SERVICE_SUCCESS.code,
                message=SERVICE_SUCCESS.msg,
                success=True,
                data=QueryRewriteResults(
                    requestId=request.requestId,
                    result=request.query,
                    needRewrite=False,
                    processTime=int((time.time() - start_time) * 1000),
                    **({} if not settings.ENABLE_SUMMARY_CHECK else {'isSummary': False, 'keywords': []})
                )
            )
            
            # 结果输出日志
            log_data = {
                "requestId": request.requestId,
                "query": request.query,
                "result": request.query,
                "needRewrite": False,
                "historyList": request.historyList if request.historyList else [],
                "processTime": int((time.time() - start_time) * 1000)
            }
            
            # 仅当总结功能开启时记录相关字段
            if settings.ENABLE_SUMMARY_CHECK:
                log_data["isSummary"] = False
                log_data["keywords"] = []
                
            formatted_message(
                logger,
                "正常响应（查询长度超限，返回原始查询）",
                log_data
            )
            return result
        
        # 创建任务列表
        tasks = []
        
        # 添加总结判断任务
        if settings.ENABLE_SUMMARY_CHECK:
            tasks.append(("summary", asyncio.to_thread(
                call_llm_for_summary_check,
                request.query,
                request.requestId
            )))
        
        # 添加查询分类任务
        if settings.ENABLE_QUERY_CLASSIFY:
            tasks.append(("classify", asyncio.to_thread(
                call_llm_for_classify,
                request.query,
                request.requestId
            )))
        
        # 添加信息抽取任务
        extracted_info = []
        if settings.ENABLE_INFO_EXTRACTION and request.enableExtraction and request.extractionRules:
            # 统一使用批量抽取方式
            tasks.append(("extraction", asyncio.to_thread(
                call_llm_for_batch_extraction,
                request.query,
                request.extractionRules,
                request.requestId
            )))
        
        # 添加判断并改写的pipeline任务
        tasks.append(("judge_rewrite", asyncio.to_thread(
            call_judge_and_rewrite_pipeline,
            request.query,
            request.historyList,
            request.requestId
        )))
        
        # 并行执行所有任务
        results = {}
        for name, task in tasks:
            results[name] = await task
        
        # 获取并处理各任务的结果
        if settings.ENABLE_SUMMARY_CHECK and "summary" in results:
            summary_result = results["summary"]
            is_summary = summary_result["isSummary"]
            keywords = summary_result["keywords"]
        else:
            is_summary = False
            keywords = []
        
        if settings.ENABLE_QUERY_CLASSIFY and "classify" in results:
            query_type = results["classify"]
        else:
            query_type = QueryType.OTHER
        
        # 获取信息抽取结果
        if settings.ENABLE_INFO_EXTRACTION and request.enableExtraction and request.extractionRules:
            if "extraction" in results and results["extraction"]:
                # 使用批量抽取的结果
                extracted_info = results["extraction"]
        
        # 获取判断改写pipeline的结果
        judge_rewrite_result = results.get("judge_rewrite", {"needRewrite": False, "result": request.query})
        need_rewrite = judge_rewrite_result["needRewrite"]
        final_query = judge_rewrite_result["result"]
        
        # 准备额外字段
        extra_fields = {}
        if settings.ENABLE_QUERY_CLASSIFY:
            extra_fields['queryType'] = query_type
        if settings.ENABLE_INFO_EXTRACTION and request.enableExtraction and request.extractionRules:
            extra_fields['extractedInfo'] = extracted_info
            
        result = QueryRewriteResponse(
            code=SERVICE_SUCCESS.code,
            message=SERVICE_SUCCESS.msg,
            success=True,
            data=QueryRewriteResults(
                requestId=request.requestId,
                result=final_query,
                needRewrite=need_rewrite,
                processTime=int((time.time() - start_time) * 1000),
                **extra_fields
            )
        )
        
        # 结果输出日志
        log_data = {
            "requestId": request.requestId,
            "query": request.query,
            "result": final_query,
            "needRewrite": need_rewrite,
            "historyList": request.historyList if request.historyList else [],
            "processTime": int((time.time() - start_time) * 1000)
        }
        
        # 仅当查询分类功能开启时记录相关字段
        if settings.ENABLE_QUERY_CLASSIFY:
            log_data["queryType"] = get_query_type_label(query_type)
            
        # 仅当信息抽取功能开启时记录相关字段
        if settings.ENABLE_INFO_EXTRACTION and request.enableExtraction:
            log_data["extractedInfo"] = extracted_info
            
        action_desc = "查询改写完成" if need_rewrite else "无需改写，返回原始查询"
        formatted_message(
            logger,
            f"正常响应（{action_desc}）",
            log_data
        )
        
        return result
        
    except Exception as e:
        error_msg = f"异常捕获（查询改写失败）: {str(e)}"
        # 错误日志
        formatted_message(
            logger,
            error_msg,
            {
                "requestId": request.requestId,
                "query": request.query,
                "error": str(e)
            }
        )
        raise CustomException(UNK_ERROR.code, error_msg)


def clean_output(text: str) -> str:
    """清理和格式化模型输出"""
    original_text = text
    processed_text = text

    # 移除多余的前缀
    prefixes = ["改写后查询：", "查询：", "问题：", "query：", "query:"]
    for prefix in prefixes:
        if processed_text.startswith(prefix):
            before_prefix_removal = processed_text
            processed_text = processed_text[len(prefix):].strip()
            formatted_message(logger, "`clean_output` - 移除前缀后", {"before": before_prefix_removal, "after": processed_text}, level="debug")

    # 移除引号
    if (processed_text.startswith('"') and processed_text.endswith('"')) or (processed_text.startswith("'") and processed_text.endswith("'")):
        before_quote_removal = processed_text
        processed_text = processed_text[1:-1].strip()
        formatted_message(logger, "`clean_output` - 移除引号后", {"before": before_quote_removal, "after": processed_text}, level="debug")

    # 移除多余空格
    before_space_removal = processed_text
    processed_text = " ".join(processed_text.split())
    if before_space_removal != processed_text:
        formatted_message(logger, "`clean_output` - 移除多余空格后", {"before": before_space_removal, "after": processed_text}, level="debug")

    # 确保返回内容非空
    if not processed_text:
        before_empty_check = processed_text
        processed_text = "查询内容为空"
        formatted_message(logger, "`clean_output` - 处理空内容后", {"before": before_empty_check, "after": processed_text}, level="debug")

    if original_text != processed_text:
        formatted_message(
            logger,
            "`clean_output` 最终结果",
            {
                "original": original_text,
                "final": processed_text
            },
            level="debug"
        )
    
    return processed_text


# 定义QueryType对应的中文标签
def get_query_type_label(query_type):
    """获取查询类型的中文标签"""
    query_type_labels = {
        0: "其他类",
        1: "总结类",
        2: "建议类",
        3: "发言稿"
    }
    return query_type_labels.get(query_type, "未知类型")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "version": settings.version}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=settings.main_service_host,
        port=int(settings.main_service_port),
        reload=(settings.env == 'dev')  # 仅在开发环境下启用热重载
    )