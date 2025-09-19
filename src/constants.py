# Date    : 2024/7/17 17:05
# File    : constants.py
# Desc    : 查询改写服务常量定义
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import enum
from enum import Enum, IntEnum


class SERVICE_SUCCESS(object):
    code = "0000"
    msg = "Service response successful"


class UNK_ERROR(object):
    code = "9999"
    msg = "Unknown error"


class REQ_PARAM_ERROR(object):
    code = "01000001"
    msg = "Request parameter error"


class SERVICE_ERROR(object):
    code = "01000101"
    msg = "Internal service response exception"


DEV_TEST_MODE = {'test', 'dev'}


# 定义一个枚举类型来表示应用的状态
class AppStatus(str, Enum):
    LOADING_MODEL = "LOADING_MODEL"
    MODEL_LOADED_SUCCESS = "MODEL_LOADED_SUCCESS"
    MODEL_LOADED_FAILED = "MODEL_LOADED_FAILED"
    APP_STARTED_SUCCESS = "APP_STARTED_SUCCESS"
    APP_STARTED_FAILED = "APP_STARTED_FAILED"


# 定义一个整型枚举来表示查询类型
class QueryType(IntEnum):
    OTHER = 0      # 其他类
    SUMMARY = 1    # 总结类
    SUGGESTION = 2 # 建议类
    SPEECH = 3     # 发言稿


# API Response Codes
SUCCESS_CODE = "0000"
SYSTEM_ERROR = "9999"

# Response Messages
SUCCESS_MSG = "success"
SYSTEM_ERROR_MSG = "system error"

# Validation Constants
MAX_QUERY_LENGTH = 128
MAX_HISTORY_QUERIES = 5
MIN_QUERY_LENGTH = 1

# Model Constants
JUDGE_MODEL_NAME = "qwen2.5-1.5b"
REWRITE_MODEL_NAME = "qwen2.5-3b"
SUMMARY_MODEL_NAME = "qwen2.5-3b"  # 总结判断模型名称
EXTRACTION_MODEL_NAME = "qwen2.5-3b"  # 信息抽取模型名称

# API Routes
API_PREFIX = "/yun/ai/text"
QUERY_REWRITE_ROUTE = "/query/rewrite"

# Validation Messages
QUERY_LENGTH_ERROR = "Query length must be between 1 and 128 characters"
HISTORY_LENGTH_ERROR = "History queries must not exceed 5 items"
QUERY_CONTENT_ERROR = "Query must not be empty or only whitespace"
INVALID_REQUEST_ID = "Invalid request ID"

# Processing
DEFAULT_TIMEOUT = 10  # seconds
MAX_RETRY_ATTEMPTS = 3

# Prompt Files
JUDGE_PROMPTS_FILE = "judge_prompts.yaml"
REWRITE_PROMPTS_FILE = "rewrite_prompts.yaml"
SUMMARY_PROMPTS_FILE = "summary_prompts.yaml"  # 总结判断提示词文件
CLASSIFY_PROMPTS_FILE = "classify_prompts.yaml"  # 查询分类提示词文件
EXTRACTION_PROMPTS_FILE = "extraction_prompts.yaml"  # 信息抽取提示词文件

# 提示词版本在settings.py中配置，这里不再定义
# JUDGE_PROMPT_VERSION = "v1"
# REWRITE_PROMPT_VERSION = "v1"
# SUMMARY_PROMPT_VERSION = "v1"  # 总结判断提示词版本
