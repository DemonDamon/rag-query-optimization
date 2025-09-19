# Date    : 2024/7/17 17:05
# File    : constants.py
# Desc    :
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import enum
from enum import Enum


SUCCESS_CODE = "0000"
SUCCESS_MSG = "服务响应成功"


class UNK_ERROR(object):
    code = "9999"
    msg = "未知错误"


class REQ_PARAM_ERROR(object):
    code = "01000001"
    msg = "请求参数错误"


class SERVICE_ERROR(object):
    code = "01000101"
    msg = "内部服务响应异常"


class ServerMode(str, enum.Enum):
    DEV_MODE = "dev"  # 开发模式
    TEST_MODE = "test"  # 测试模式
    PRODUCTION_MODE = "prod"  # 生产模式


# 定义一个枚举类型来表示应用的状态
class AppStatus(str, Enum):
    LOADING_MODEL = "LOADING_MODEL"
    MODEL_LOADED_SUCCESS = "MODEL_LOADED_SUCCESS"
    MODEL_LOADED_FAILED = "MODEL_LOADED_FAILED"
    APP_STARTED_SUCCESS = "APP_STARTED_SUCCESS"
    APP_STARTED_FAILED = "APP_STARTED_FAILED"
