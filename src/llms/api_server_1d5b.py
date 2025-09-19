# Date    : 2024/7/9 15:50
# File    : qwen_api_server.py
# Desc    :
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import os
import sys
import json
import time

# 添加项目根目录
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

import argparse
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import uuid
import uvicorn

from pydantic import BaseModel, Field
from fastapi import FastAPI
import threading
import traceback

import torch
from typing import Dict, List, Literal, Optional, Union

from src.llms.qwen import Qwen2Local
from src.llms.constants import AppStatus
from src.llms.sse import ServerSentEvent, EventSourceResponse

from logger import logger, formatted_message


"""
cli: 
VLLM_USE_PRECOMPILED=1 VLLM_SKIP_KERNEL_COMPILATION=1 python api_server_qwen.py --model_name_or_path /data/huggingface_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1/ --use_vllm true --vllm_gpu_mem_util 0.8 --port 12311
"""

parser = argparse.ArgumentParser(description="Run the model server client.")

# 添加模型路径参数
parser.add_argument("--model_name_or_path", type=str, required=True,
                    help="Path to pre-trained model or shortcut name.")

# 添加lora路径参数
parser.add_argument("--lora_path", type=str, default=None,
                    help="Path to lora model.")

# 添加是否使用vllm参数
parser.add_argument("--use_vllm", type=str, required=False,
                    help="Whether to use vllm or not. (True or False)")

# 添加是否使用vllm启动占用GPU显存比例
parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.9,
                    help="the rate of using vllm gpu mem. (default=0.9)")

# 添加主机地址参数
parser.add_argument("--host", type=str, default="0.0.0.0",
                    help="Host address of the model server.")

# 添加端口号参数
parser.add_argument("--port", type=int, default=12311,
                    help="Port number of the model server.")

# 添加日志路径参数
parser.add_argument("--logdir", type=str, default="multitask.log",
                    help="Path to the log file.")

# 添加是否使用mock参数
parser.add_argument("--mock", action="store_true", help="Whether to use mock data or not.")

args = parser.parse_args()

# 定义一个全局变量来存储应用的状态
app_status = None

# 添加请求锁，避免并发问题
request_lock = threading.Lock()

if args.mock:
    # 这里执行使用模拟数据的逻辑
    logger.info("Using mock data...")
else:
    # 这里执行正常的逻辑
    logger.info("Using real data...")
    
    # 修改检查依赖的逻辑，分别检查每个命令
    try:
        if args.use_vllm and args.use_vllm.lower() == "true":
            # 分别检查ccache和gcc是否可用
            ccache_available = os.system("which ccache > /dev/null 2>&1") == 0
            gcc_available = os.system("which gcc > /dev/null 2>&1") == 0
            
            if not ccache_available:
                logger.warning("ccache not found, this might cause issues with VLLM")
            if not gcc_available:
                logger.warning("gcc not found, this might cause issues with VLLM")
                
            # 即使依赖检查失败，也继续尝试启动
            logger.info("Continuing with VLLM initialization regardless of dependency check results")
    except Exception as e:
        logger.warning(f"Failed to check dependencies: {str(e)}")
        logger.warning("Continuing with initialization regardless of dependency check failure")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global app_status

    # 启动事件
    try:
        await init_model()
        app_status = AppStatus.APP_STARTED_SUCCESS  # 设置状态为"启动服务成功"
    except Exception as e:
        logger.error(f"Failed to initialize app: {str(e)}")
        logger.error(traceback.format_exc())
        app_status = AppStatus.APP_STARTED_FAILED  # 设置状态为"启动服务失败"

    yield

    # 退出时清理资源
    try:
        logger.info("Cleaning up resources...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


app = FastAPI(lifespan=lifespan)


# 全局变量来存储模型实例
model = None
model_name = args.model_name_or_path  # "Qwen/Qwen2-7B-Instruct"
use_vllm = args.use_vllm
vllm_gpu_mem_util = args.vllm_gpu_mem_util


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


# 清理GPU资源
def clean_gpu_resources():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error clearing GPU cache: {str(e)}")


class RequestModel(BaseModel):
    request_id: str
    input: str
    history: Optional[List[Dict]] = []
    max_new_token: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    streaming: Optional[bool] = False


class ResponseData(BaseModel):
    request_id: str
    result: str


class ResponseModel(BaseModel):
    success: bool = True
    code: str = "0000"
    message: str = "success"
    data: ResponseData


class Message(BaseModel):
    role: Union[Literal["user", "assistant", "system"], None]
    content: Union[str, None]


class ApiChatCompletionRequest(BaseModel):
    requestId: str
    messages: List[Message]
    # model: str = None
    # temperature: Optional[float] = 0.7
    # topP: Optional[float] = 1.0
    # n: Optional[int] = 1
    # maxGenerateTokens: Optional[int] = None
    # maxInputTokens: Optional[int] = 6000
    # seed: Optional[int] = None
    # stream: Optional[bool] = False


class UsageInfo(BaseModel):
    promptTokens: int = 0
    totalTokens: int = 0
    completionTokens: Optional[int] = 0


class ApiChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finishReason: Optional[Literal["stop", "length"]] = None


class ApiChatCompletionResponse(BaseModel):
    requestId: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    choices: List[ApiChatCompletionResponseChoice]
    usage: UsageInfo = None


class ResponseModelV1(BaseModel):
    success: bool = True
    code: str = "0000"
    message: str = "success"
    data: ApiChatCompletionResponse


async def init_model():
    global model, app_status

    app_status = AppStatus.LOADING_MODEL

    if not args.mock:
        try:
            if "qwen" in model_name.lower():
                # 设置环境变量以防止可能的问题
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                
                # 初始化模型
                model = Qwen2Local(model_name,
                                   use_vllm=use_vllm,
                                   vllm_gpu_mem_util=vllm_gpu_mem_util,
                                   lora_path=args.lora_path)
                app_status = AppStatus.MODEL_LOADED_SUCCESS
            else:
                raise ValueError("Model name not supported")

            logger.debug(f"{model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            logger.error(traceback.format_exc())
            app_status = AppStatus.MODEL_LOADED_FAILED
            
            # 记录模型初始化失败详细日志
            formatted_message(
                logger,
                msg="【实体抽取】大模型服务：模型初始化失败",
                reqs_info={
                    "model_name": model_name,
                    "error": str(e)
                }
            )

            raise ValueError("Failed to initialize model")
    else:
        model = "mock"


@app.post("/yun/ai/internal/llm/qwen/1d5b")
async def inference(request: RequestModel):
    global model, request_lock

    # 记录请求开始日志
    formatted_message(
        logger,
        msg="【实体抽取】大模型服务：请求开始处理",
        reqs_info={
            "request_id": request.request_id,
            "input_length": len(request.input) if request.input else 0,
            "history_length": len(request.history) if request.history else 0,
            "max_new_token": request.max_new_token,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "streaming": request.streaming
        }
    )

    if args.mock:
        result = """[{\"intention\": \"012\",\"score\": 0.0,\"entityList\": [{\"timeList\": [\"昨天\"],\"placeList\": [\"广州\"],\"labelList\": [],\"metaDataList\": [{\"key\": \"1\",\"value\": [\"广州塔\"]}],\"imageNameList\": []}]}]"""
        return ResponseModel(
            data=ResponseData(
                request_id=request.request_id,
                result=result
            )
        )

    if model is None:
        formatted_message(
            logger,
            msg="【实体抽取】大模型服务：模型初始化失败",
            reqs_info={
                "request_id": request.request_id,
                "error": "Model initialization failed"
            }
        )
        return ResponseModel(
                success=False,
                code="9999",
                message="Model initialization failed",
                data=ResponseData(
                    request_id=request.request_id,
                    result=""
                )
            )

    # 使用线程锁来避免并发问题
    with request_lock:
        try:
            # 记录请求信息
            logger.debug(f"Processing request: {request.request_id}")
            
            # 记录使用的参数
            logger.info(f"Request parameters: temperature={request.temperature}, top_p={request.top_p}, max_new_token={request.max_new_token}, streaming={request.streaming}")
            
            # 对于流式响应
            if request.streaming:
                async def event_generator():
                    try:
                        import asyncio
                        import time
                        
                        # 对于VLLM模型的流式处理
                        if use_vllm and hasattr(model, "model") and hasattr(model.model, "generate_stream"):
                            from vllm import SamplingParams
                            
                            # 确保温度是正数
                            temperature = request.temperature
                            if temperature <= 0:
                                logger.warning(f"Temperature must be positive, got {temperature}. Using 0.01 instead.")
                                temperature = 0.01
                            
                            # 创建新的采样参数
                            sampling_params = SamplingParams(
                                temperature=temperature,
                                top_p=request.top_p,
                                repetition_penalty=1.0,
                                max_tokens=request.max_new_token,
                                seed=42,  # 确保可重现性
                                stop=None,  # 停止词列表
                                ignore_eos=False  # 是否忽略EOS token
                            )
                            
                            # 准备输入
                            if len(request.history) == 0:
                                messages = [
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": request.input}
                                ]
                            else:
                                messages = request.history + [{"role": "user", "content": request.input}]
                            
                            text_input = model.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            
                            # 提取结果
                            full_text = ""
                            created_time = int(time.time())
                            # 使用VLLM的流式API
                            for output in model.model.generate_stream([text_input], sampling_params):
                                if output.outputs and len(output.outputs) > 0:
                                    current_output = output.outputs[0].text
                                    new_text = current_output[len(full_text):]
                                    if new_text:
                                        logger.debug(f"Streaming chunk: {new_text}")
                                        
                                        # 创建OpenAI兼容的响应格式
                                        response_data = {
                                            "choices": [
                                                {
                                                    "delta": {
                                                        "content": new_text,
                                                        "role": "assistant"
                                                    },
                                                    "index": 0,
                                                    "logprobs": None,
                                                    "finish_reason": None
                                                }
                                            ],
                                            "object": "chat.completion.chunk",
                                            "usage": None,
                                            "created": created_time,
                                            "model": model_name,
                                            "id": request.request_id
                                        }
                                        
                                        yield ServerSentEvent(data=json.dumps(response_data))
                                        full_text = current_output
                                        await asyncio.sleep(0.01)
                            
                            # 发送完成事件
                            finish_data = {
                                "choices": [
                                    {
                                        "delta": {},
                                        "index": 0,
                                        "logprobs": None,
                                        "finish_reason": "stop"
                                    }
                                ],
                                "object": "chat.completion.chunk",
                                "usage": None,
                                "created": created_time,
                                "model": model_name,
                                "id": request.request_id
                            }
                            yield ServerSentEvent(data=json.dumps(finish_data))
                                        
                        else:
                            # 非VLLM模型使用模拟流式传输
                            import asyncio
                            
                            # 首先生成完整的回复
                            logger.debug("Using simulated streaming for non-VLLM model")
                            result = model.invoke(
                                request.input,
                                history=request.history,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                max_new_tokens=request.max_new_token
                            )
                            
                            # 使用请求中的request_id保持一致性
                            created_time = int(time.time())
                            
                            # 模拟流式输出，分块发送
                            chunk_size = 1  # 每次只发送一个字符
                            for i in range(0, len(result), chunk_size):
                                chunk = result[i:i+chunk_size]
                                if chunk:
                                    logger.debug(f"Sending OpenAI format chunk: {chunk}")
                                    
                                    # 创建OpenAI兼容的响应格式
                                    response_data = {
                                        "choices": [
                                            {
                                                "delta": {
                                                    "content": chunk,
                                                    "role": "assistant"
                                                },
                                                "index": 0,
                                                "logprobs": None,
                                                "finish_reason": None
                                            }
                                        ],
                                        "object": "chat.completion.chunk",
                                        "usage": None,
                                        "created": created_time,
                                        "model": model_name,
                                        "id": request.request_id
                                    }
                                    
                                    yield ServerSentEvent(data=json.dumps(response_data))
                                    await asyncio.sleep(0.05)
                            
                            # 发送完成事件
                            finish_data = {
                                "choices": [
                                    {
                                        "delta": {},
                                        "index": 0,
                                        "logprobs": None,
                                        "finish_reason": "stop"
                                    }
                                ],
                                "object": "chat.completion.chunk",
                                "usage": None,
                                "created": created_time,
                                "model": model_name,
                                "id": request.request_id
                            }
                            yield ServerSentEvent(data=json.dumps(finish_data))
                            
                    except Exception as e:
                        logger.error(f"流式生成错误: {str(e)}")
                        logger.error(traceback.format_exc())
                        
                        # 记录流式生成错误日志
                        formatted_message(
                            logger,
                            msg="【实体抽取】大模型服务：流式生成错误",
                            reqs_info={
                                "request_id": request.request_id,
                                "error": str(e)
                            }
                        )
                        
                        # 错误信息也使用标准格式
                        error_data = {
                            "success": False,
                            "code": "9999",
                            "message": str(e),
                            "data": {
                                "request_id": request.request_id,
                                "result": ""
                            }
                        }
                        yield ServerSentEvent(data=json.dumps(error_data))
                    finally:
                        # 清理资源
                        clean_gpu_resources()
                
                # 返回流式响应
                return EventSourceResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    ping=5  # 每5秒发送一次ping保持连接
                )
            else:
                # 对于VLLM模型，更新采样参数
                if use_vllm and hasattr(model, "model") and hasattr(model, "sampling_params"):
                    from vllm import SamplingParams
                    
                    # 记录请求开始时间
                    st_request = time.time()
                    
                    # 确保温度是正数
                    temperature = request.temperature
                    if temperature <= 0:
                        logger.warning(f"Temperature must be positive, got {temperature}. Using 0.01 instead.")
                        temperature = 0.01
                    
                    # 创建新的采样参数，遵循vllm最新API规范
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        top_p=request.top_p,
                        repetition_penalty=1.0,
                        max_tokens=request.max_new_token,
                        seed=42,  # 确保可重现性
                        stop=None,  # 停止词列表
                        ignore_eos=False  # 是否忽略EOS token
                    )
                    
                    # 调用模型处理请求
                    outputs = model.model.generate(
                        [model.tokenizer.apply_chat_template(
                            [{"role": "user", "content": request.input}],
                            tokenize=False,
                            add_generation_prompt=True
                        )], 
                        sampling_params
                    )
                    
                    # 提取结果
                    result = ""
                    for output in outputs:
                        result += output.outputs[0].text
                else:
                    # 记录请求开始时间
                    st_request = time.time()
                
                    # 对于非VLLM模型，使用标准调用，传递所有参数
                    result = model.invoke(
                        request.input,
                        history=request.history,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_new_tokens=request.max_new_token
                    )
                
                # 请求成功完成后清理资源
                clean_gpu_resources()
                
                # 记录请求成功完成日志
                formatted_message(
                    logger,
                    msg="【实体抽取】大模型服务：请求处理成功",
                    reqs_info={
                        "request_id": request.request_id,
                        "result_length": len(result) if result else 0,
                        "process_time": f"{int((time.time() - st_request) * 1000)}ms" if 'st_request' in locals() else "unknown"
                    }
                )
                
                return ResponseModel(
                    data=ResponseData(
                        request_id=request.request_id,
                        result=result
                    )
                )
        except Exception as e:
            # 记录详细的错误信息
            logger.error(f"Error during inference: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 记录异常日志
            formatted_message(
                logger,
                msg="【实体抽取】大模型服务：请求处理异常",
                reqs_info={
                    "request_id": request.request_id,
                    "error": str(e)
                }
            )
            
            # 尝试清理资源
            clean_gpu_resources()
            
            return ResponseModel(
                success=False,
                code="9999",
                message=str(e),
                data=ResponseData(
                    request_id=request.request_id,
                    result=""
                )
            )


# 添加健康检查端点
@app.get("/health")
async def health_check():
    global app_status
    return {
        "status": "ok" if app_status == AppStatus.APP_STARTED_SUCCESS else "error",
        "app_status": app_status,
        "model": model_name,
        "use_vllm": use_vllm
    }


if __name__ == "__main__":
    logger.add(args.logdir, rotation="500 MB")
    uvicorn.run(app, host=args.host, port=args.port)
