# Date    : 2024/7/22 13:37
# File    : settings.py
# Desc    : 查询改写服务配置
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import os
import sys
import json
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 添加项目根目录
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from src.utils import Computer


# 获取当前py文件的路径
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)

ENV = os.environ.get('ENV')
ENV_FILE = os.path.join(current_dir_path, f'.{ENV}.env')
assert ENV_FILE, 'env_file not found'


class Settings(BaseSettings):
    """
    1. 允许读取.env静态配置文件，该文件一般存储敏感信息（如数据库密码、API秘钥等）以及一些不常变动的配置
    2. 读取json配置文件

    示例：
    print(Settings.from_json('settings.json'))
    """

    # 读取.env静态配置文件，并设置环境变量大小写敏感
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding='utf-8', extra='allow')
    # model_config = SettingsConfigDict(
    #     env_file=ENV_FILE,
    #     env_file_encoding='utf-8',
    #     extra='allow',
    #     case_sensitive_env=True  # Pydantic V2中使用case_sensitive_env替代case_sensitive
    # )

    @classmethod
    def from_json(cls, json_path: Path):
        assert os.path.exists(json_path), f'File {json_path} does not exist.'
        with open(json_path, 'r') as f:
            data = json.load(f)

        return cls(**data)


settings = Settings()
settings.version = '1.0.0'
settings.env = ENV
# settings.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

settings.main_service_url = f'{settings.protocol}://{settings.main_service_host}:{settings.main_service_port}{settings.main_service_endpoint}'


# 日志配置
settings.logger_name = "query-rewrite"

if ENV == 'dev':
    settings.log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    settings.log_sink = f"{settings.log_dir}/logs/{settings.logger_name}.log"
else:
    settings.log_dir = f"/logs/{settings.logger_name}/{Computer.NAME}"
    settings.log_sink = f"{settings.log_dir}/{settings.logger_name}.log"

settings.log_level_stdout = logging.DEBUG
settings.logger_format_stdout = "{time} | {level} | {name}:{function}:{line} - {message}"

settings.log_level_file = logging.INFO
settings.logger_format_file = "{time:YYYY-MM-DD HH:mm:ss.SSS}" + \
                             f"|{Computer.IP}|{Computer.NAME}|{settings.service_name}" + \
                              "|p{process}d|t{thread.name}d|{level}|API|||{message}|"

# 业务配置

# 提示词配置
settings.JUDGE_PROMPTS_FILE = os.path.join(current_dir_path, "judge_prompts.yaml")
settings.REWRITE_PROMPTS_FILE = os.path.join(current_dir_path, "rewrite_prompts.yaml")
settings.SUMMARY_PROMPTS_FILE = os.path.join(current_dir_path, "summary_prompts.yaml")
settings.CLASSIFY_PROMPTS_FILE = os.path.join(current_dir_path, "classify_prompts.yaml")
settings.EXTRACTION_PROMPTS_FILE = os.path.join(current_dir_path, "extraction_prompts.yaml")

settings.PROMPT_VERSION = "v1"

# 分别定义judge和rewrite的版本号
settings.JUDGE_PROMPT_VERSION = "v2"
settings.REWRITE_PROMPT_VERSION = "v2"
settings.SUMMARY_PROMPT_VERSION = "v1"  # 使用v3版本提示词
settings.CLASSIFY_PROMPT_VERSION = "v1"  # 查询分类提示词版本
settings.EXTRACTION_PROMPT_VERSION = "v3"  # 信息抽取提示词版本

# 大模型配置
settings.JUDGE_MODEL_NAME = "qwen2.5-1.5b"
settings.REWRITE_MODEL_NAME = "qwen2.5-3b"
settings.SUMMARY_MODEL_NAME = "qwen2.5-3b"
settings.CLASSIFY_MODEL_NAME = "qwen2.5-3b"  # 查询分类模型
settings.EXTRACTION_MODEL_NAME = "qwen2.5-3b"  # 信息抽取模型

# 模型调用参数
settings.MAX_NEW_TOKEN = 256  # 查询场景不需要太多token
settings.TEMPERATURE = 0.01    # 低温度保证确定性输出
settings.TOP_P = 0.9
settings.TIMEOUT = 10         # 请求超时时间（秒）

# 功能开关
settings.ENABLE_SUMMARY_CHECK = False  # 是否启用总结判断功能，默认关闭
settings.ENABLE_STEP_BACK_PROMPTING = False  # 是否启用Step-Back Prompting功能，默认关闭
settings.ENABLE_QUERY_CLASSIFY = True  # 是否启用查询分类功能，默认开启
settings.ENABLE_INFO_EXTRACTION = True  # 是否启用信息抽取功能，默认开启