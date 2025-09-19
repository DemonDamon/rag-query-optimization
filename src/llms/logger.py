# Date    : 2024/7/27 11:40
# File    : logger.py
# Desc    : 
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import os
import sys
import logging
from types import FrameType
from typing import cast, Union, Dict
import socket

from loguru import logger
from settings import settings


ENV = os.getenv('ENV')

def get_host_ip():
    ip = ''
    host_name = ''
    try:
        sc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sc.connect(('8.8.8.8', 80))
        ip = sc.getsockname()[0]
        host_name = socket.gethostname()
        sc.close()
    except Exception:
        pass
    return ip, host_name


# 时间|ip地址|主机名|服务名|进程|线程|日志级别|日志内容|请求信息|
computer_ip, computer_name = get_host_ip()

# 设置日志的名称 (loguru 支持通过调用 logger.bind 方法来附加额外的上下文信息)
logger = logger.bind(name=settings.logger_name)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("uvicorn"):
            return
        
        formatted_message(logger, record.getMessage())


class NoLogInterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        pass


def formatted_message(loguru_obj, msg: Union[str, list] = None, reqs_info: Dict = {}, level: str = "info"):
    """
    logger设置info级别流入标准elk日志；debug级别日志流入自定义日志路径方便调试

    loguru_obj：loguru对象
    msg：具体日志信息
    reqs_info：请求信息
    level：日志级别，默认为info
    """
    if msg:
        # 请求信息部分
        _reqs_parts = []
        for _key, _value in reqs_info.items():
            _reqs_parts.append(f"{str(_key)}={str(_value)}")
        _reqs_parts_msg = "|".join(_reqs_parts)

        if isinstance(msg, list):
            for _msg in msg:
                if _msg:
                    _tmp = f"{str(_msg)}|{_reqs_parts_msg}" if _reqs_parts_msg else str(_msg)
                    getattr(loguru_obj, level)(_tmp)
        else:
            _tmp = f"{str(msg)}|{_reqs_parts_msg}" if _reqs_parts_msg else str(msg)
            getattr(loguru_obj, level)(_tmp)


def init_logger():
    # 设置 uvicorn 日志处理
    logging.getLogger("uvicorn").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]

    # 其他日志处理保持不变
    for _name in ["python.nn_module", "huggingface_hub.hub_mixin"]:
        logging.getLogger(_name).handlers = [NoLogInterceptHandler()]

    # 移除所有现有的 handlers
    logger.remove()

    # 添加文件日志 handler
    logger.add(
        settings.log_sink,
        level=settings.log_level_file,
        format=settings.logger_format_file,
        rotation='5 MB'
    )

    # 在开发环境下添加控制台日志
    if settings.env == 'dev':
        logger.add(
            sys.stdout,
            level=settings.log_level_stdout,
            filter=lambda record: record["level"].name in ["DEBUG", "ERROR", "WARNING"]
        )

    if not os.path.exists(settings.log_dir):
        os.makedirs(settings.log_dir, exist_ok=True)

    return logger


# 初始化日志
logger = init_logger()
