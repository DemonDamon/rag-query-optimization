import json
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm


"""
=== Statistics ===
Total samples: 12686
Average text length: 10377.41 chars
Average token count: 6306.99 tokens
Max token count: 6481 tokens
Min token count: 6280 tokens
Token count std: 28.94 tokens
50th percentile: 6296.00 tokens
90th percentile: 6348.00 tokens
95th percentile: 6369.00 tokens
99th percentile: 6407.00 tokens
"""


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/data/codes/intent-recognition/weights/qwen2_7b_chat_ft_v18_mcloud', trust_remote_code=True)
    
    # 读取数据文件
    data_file = '/data/codes/thirdparty/LLaMA-Factory-All-Release/v0.8.0/data/intents_v16_20241124.json'  # 根据实际路径调整
    data = read_json(data_file)
    
    # 存储每条数据的token数量
    token_counts = []
    char_counts = []
    
    # 创建输出文件
    with open('token_statistics.txt', 'w', encoding='utf-8') as f:
        # 遍历每条数据（添加进度条）
        for idx, item in tqdm(enumerate(data), total=len(data), desc="处理样本"):
            # 获取instruction文本
            text = f"{item['instruction']}{item['output']}"
            
            # 计算token数量
            token_ids = tokenizer.encode(text)
            token_count = len(token_ids)
            char_count = len(text)
            
            # 保存结果
            token_counts.append(token_count)
            char_counts.append(char_count)
            
            # 写入详细信息
            f.write(f"Sample {idx+1}:\n")
            f.write(f"Total text len: {char_count}\n")
            f.write(f"Total tokens: {token_count}\n\n")
        
        # 计算并写入统计信息
        f.write("\n=== Statistics ===\n")
        f.write(f"Total samples: {len(data)}\n")
        f.write(f"Average text length: {np.mean(char_counts):.2f} chars\n")
        f.write(f"Average token count: {np.mean(token_counts):.2f} tokens\n")
        f.write(f"Max token count: {max(token_counts)} tokens\n")
        f.write(f"Min token count: {min(token_counts)} tokens\n")
        f.write(f"Token count std: {np.std(token_counts):.2f} tokens\n")
        
        # 打印百分位数信息
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(token_counts, p)
            f.write(f"{p}th percentile: {value:.2f} tokens\n")


if __name__ == "__main__":
    main()
