#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import yaml
import requests
import json
from datetime import datetime
import time
from tqdm import tqdm

# API配置
API_URL = "https://api.gptsapi.net/v1/chat/completions"
API_KEY = "sk-Xwef82847d1ff76caf4f9ee1bb554dcb1eaf52f2ed106m9s"
MODEL = "claude-3-5-sonnet-20241022"  # 使用Claude 3.5 Sonnet
# MODEL = "claude-3-5-haiku-20241022"  # 使用Claude 3.5 haiku
EXCEL_FILE_NAME = "微调数据.xlsx"


def load_yaml_prompt(file_path, version="v1"):
    """
    加载YAML文件中的提示词
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
            return prompts.get(version, "")
    except Exception as e:
        print(f"加载提示词文件失败: {e}")
        return ""

def call_claude_api(prompt, max_retries=3, retry_delay=2):
    """
    调用Claude API
    """
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            return content.strip()
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None

def format_history_queries(row):
    """
    格式化历史查询，跳过空值
    """
    history = []
    for i in range(1, 6):  # Q1到Q5
        col_name = f'Q{i}'
        if col_name in row and pd.notna(row[col_name]) and row[col_name]:
            history.append(f"Q{i}: {row[col_name]}")
    return "\n".join(history)

def judge_need_rewrite(judge_prompt_template, row):
    """
    判断是否需要改写
    """
    # 获取历史查询
    history_text = format_history_queries(row)
    if not history_text:
        return False  # 没有历史查询，不需要改写
    
    # 构建完整提示词
    prompt = judge_prompt_template.format(
        history_queries=history_text,
        current_query=row['CurrentQuery']
    )
    
    # 调用API
    result = call_claude_api(prompt)
    
    # 解析结果
    if result and result.lower() == 'true':
        return True
    else:
        return False

def rewrite_query(rewrite_prompt_template, row):
    """
    改写查询
    """
    # 跳过不需要改写或已经有预期查询的行
    if not row['NeedRewrite'] or pd.notna(row['ExpectedQuery']) or pd.isna(row['CurrentQuery']):
        return row['ExpectedQuery']
    
    # 获取历史查询
    history_text = format_history_queries(row)
    if not history_text:
        return row['CurrentQuery']  # 没有历史查询，返回原始查询
    
    # 构建完整提示词
    prompt = rewrite_prompt_template.format(
        history_text=history_text,
        current_query=row['CurrentQuery']
    )
    
    # 调用API
    result = call_claude_api(prompt)
    
    return result if result else row['CurrentQuery']

def process_with_progress(df, func, prompt_template, desc):
    """
    使用进度条处理DataFrame
    """
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        result = func(prompt_template, row)
        results.append(result)
    return results

def main():
    # 当前脚本路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 文件路径
    excel_path = os.path.join(current_dir, EXCEL_FILE_NAME)
    judge_prompts_path = os.path.join(parent_dir, "src", "judge_prompts.yaml")
    rewrite_prompts_path = os.path.join(parent_dir, "src", "rewrite_prompts.yaml")
    
    # 读取Excel文件
    try:
        df = pd.read_excel(excel_path)
        print(f"成功读取Excel文件，包含 {len(df)} 行数据")
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return
    
    # 加载提示词模板
    judge_prompt_template = load_yaml_prompt(judge_prompts_path, "v1")
    rewrite_prompt_template = load_yaml_prompt(rewrite_prompts_path, "v1")
    
    if not judge_prompt_template or not rewrite_prompt_template:
        print("加载提示词模板失败，请检查文件路径和内容")
        return
    
    # 标记是否需要改写（使用进度条）
    print("开始判断是否需要改写...")
    df['NeedRewrite'] = process_with_progress(df, judge_need_rewrite, judge_prompt_template, "判断改写进度")
    
    # 执行查询改写（使用进度条）
    print("开始执行查询改写...")
    df['ExpectedQuery'] = process_with_progress(df, rewrite_query, rewrite_prompt_template, "查询改写进度")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(current_dir, f"微调数据_已标注_{timestamp}.xlsx")
    df.to_excel(output_path, index=False)
    print(f"处理完成，结果已保存至 {output_path}")

if __name__ == "__main__":
    main()