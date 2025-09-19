# Date    : 2024/7/9 11:31
# File    : test_qwen_local_dev.py
# Desc    : 
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import os
import sys

# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 获取 index 目录的路径
index_path = os.path.dirname(os.path.dirname(current_path))
# 将 index 目录添加到 sys.path 第一个位置
sys.path.insert(0, index_path)

from IPython.display import display, HTML
from IPython.display import clear_output
import time

from langserve import add_routes

from Qwen2 import Qwen2Local


ckp_abs_path = "D:\\myWorks\\richinfo\\codes\\intent-recognition\\weights\\qwen\\Qwen2-0.5B"
llm = Qwen2Local(ckp_abs_path)
# llm = Qwen2Local("Qwen/Qwen2-7B-Instruct")


def stream_display_in_jupyter(query):
    def _stream_data(_):
        for result in llm.stream(_):  # 假设 llm.stream 返回一个迭代器
            yield result

    def _print_stream(_):
        output = ""
        for data in _stream_data(_):
            output += data
            clear_output(wait=True)
            display(HTML(f"<pre>{output}</pre>"))
            time.sleep(0.05)  # 添加一个小延迟，使输出更平滑

    _print_stream(query)


def stream_data(query):
    for result in llm.stream(query):  # 假设 llm.stream 返回一个迭代器
        yield result


# 使用这个生成器函数
for data in stream_data("你是谁"):
    print(data)
