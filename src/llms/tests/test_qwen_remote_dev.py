# Date    : 2024/7/9 11:13
# File    : test_qwen_remote.py
# Desc    : 
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


def test_113_server():
    """有健部署"""
    import requests
    import json

    url = "http://192.168.32.113:7820/aibox/v1/llm/chat/completions"

    payload = json.dumps({
        "request_id": "test-jbjsax-89ujwbjdq-dbjdh8",
        "model": "Qwen1.5-32B-Chat-GPTQ-Int4",
        "messages": [
            {
                "role": "system",
                "content": "你是文员，负责提取客户提供的信息中的地址，地址输出要求按照以下格式，不要输出多余内容：xx省xx市xx区，成功：xx。如果提取到信息，请不要虚构，提示未找到地址"
            },
            {
                "role": "user",
                "content": "我的地址是广东省广州市天河区，可以办理宽带吗"
            }
        ],
        "stream": False,
        "max_tokens": 52
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


def test_119_server_without_history():
    """我部署"""
    import requests
    import json

    url = "http://192.168.32.119:12308/v1/llms"

    payload = json.dumps({
        "request_id": "123",
        "input": "五条悟帅吗",
        "max_new_token": 50,
        "temperature": 0.2,
        "top_p": 0.9,
        "history": [],
        "streaming": False
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


def test_119_server_with_history():
    """我部署"""
    import requests
    import json

    url = "http://192.168.32.119:12308/v1/llms"

    payload = json.dumps({
        "request_id": "123",
        "input": "我第一个问题是啥？",
        "max_new_token": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "history": [
            {
                "role": "system",
                "content": "你是一个计算器"
            },
            {
                "role": "user",
                "content": "1+1等于多少"
            },
            {
                "role": "assistant",
                "content": "1+1等于2"
            }
        ],
        "streaming": False
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
