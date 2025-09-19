# Date    : 2024/7/22 13:37
# File    : settings.py
# Desc    : 查询改写服务配置
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import os
import json
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Computer(object):
    import socket

    IP = ''
    NAME = ''
    # noinspection PyBroadException
    try:
        sc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sc.connect(('8.8.8.8', 80))
        IP = sc.getsockname()[0]
        NAME = socket.gethostname()
        sc.close()
    except Exception as e:
        pass


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
settings.env = ENV


# 日志配置
settings.logger_name = "query-rewrite"
settings.service_name = ""

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