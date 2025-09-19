# Date    : 2024/7/17 17:04
# File    : exceptions.py
# Desc    : 查询改写服务异常处理
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from models import QueryRewriteResponse, QueryRewriteResults
from constants import REQ_PARAM_ERROR


class CustomException(Exception):
    def __init__(self, code: str, msg: str):
        self.code = code
        self.msg = msg


async def custom_exception_handler(request: Request, exc: CustomException):
    """自定义异常处理器"""
    return JSONResponse(
        status_code=200,
        content=QueryRewriteResponse(
            code=exc.code,
            message=exc.msg,
            success=False,
            data=QueryRewriteResults(
                requestId=request.state.request_id if hasattr(request.state, 'request_id') else "unknown",
                result="",
                needRewrite=False,
                processTime=0
            )
        ).model_dump()
    )


async def req_valid_exception_handler(request: Request, exc: RequestValidationError):
    """请求参数验证异常处理器"""
    # 获取详细的错误信息
    error_msgs = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error["loc"])
        msg = error["msg"]
        error_msgs.append(f"{loc}: {msg}")
    
    error_msg = "; ".join(error_msgs)
    
    response_data = QueryRewriteResponse(
        code=REQ_PARAM_ERROR.code,
        message=f"请求参数错误: {error_msg}",
        success=False,
        data=QueryRewriteResults(
            requestId=request.state.request_id if hasattr(request.state, 'request_id') else "unknown",
            result="",
            needRewrite=False,
            processTime=0
        )
    )
    
    return JSONResponse(
        status_code=200,  # 返回200状态码，错误信息在响应体中
        content=response_data.model_dump()
    )
