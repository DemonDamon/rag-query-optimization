# 服务启动方式
- 默认端口是12310
- 启动路径：src/components/llms
- 启动脚本：
```bash
python api_server.py --model_name_or_path Qwen/Qwen2-7B-Instruct --use_vllm true

python api_server.py --model_name_or_path Qwen/Qwen1.5-14B-Chat --use_vllm true
```
- qwen系列都是支持的，但是可能有些还需要从hf拉模型

# 请求实例

## 不带历史对话
```python
import requests
import json

url = "http://xxx.xxx.xxx.xxx:12310/v1/llms"

payload = json.dumps({
   "request_id": "123",
   "input": "你好",
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
```

### 带历史对话

```python
import requests
import json

url = "http://xxx.xxx.xxx.xxx:12310/v1/llms"

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
```


# 微调后的启动方式
## 实体抽取
```bash
python api_server.py --model_name_or_path /data/codes/intent-recognition/weights/qwen2_1.5b_instruct_ner_ft_v1 --use_vllm true --vllm_gpu_mem_util 0.3
                                                                                 qwen2_1.5b_instruct_ner_ft_v2
```

## 意图分类
```bash
python api_server.py --model_name_or_path /data/codes/intent-recognition/weights/qwen2_7b_instruct_ft_v2 --use_vllm true --port 12311 --vllm_gpu_mem_util 0.8
                                                                                                                                                          0.9
```