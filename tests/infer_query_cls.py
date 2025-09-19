#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Date    : 2024/7/17
# File    : test_query_cls.py
# Desc    : 查询类型分类批量测试脚本
# Author  : Damon

import os
import sys
import json
import time
import uuid
import argparse
import pandas as pd
import requests
from tqdm import tqdm

# 添加项目根目录
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from src.constants import QueryType


def generate_request_id():
    """生成唯一请求ID"""
    return f"query_cls_{uuid.uuid4().hex[:8]}"


def get_label_name(query_type):
    """获取标签对应的中文名"""
    label_map = {
        QueryType.OTHER: "其他类",
        QueryType.SUMMARY: "总结类",
        QueryType.SUGGESTION: "建议类",
        QueryType.SPEECH: "发言稿"
    }
    return label_map.get(query_type, "未知类型")


def send_query_request(url, query, history_list=None):
    """发送查询请求到API"""
    if history_list is None:
        history_list = []
    
    payload = {
        "requestId": generate_request_id(),
        "query": query,
        "historyList": history_list
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"请求发送失败: {str(e)}")
        return None


def process_excel(file_path, api_url, sample_history=None):
    """处理Excel文件中的查询并发送到API"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 检查是否包含query列
        if 'query' not in df.columns:
            print("错误: Excel文件必须包含'query'列")
            return None
        
        # 添加新列
        df['pred_label'] = None
        df['label_name'] = None
        
        # 默认的历史记录样例
        if sample_history is None:
            sample_history = [
                "考试前饮食应该如何安排？",
                "这些食物能提供什么营养？",
                "奶酪的发源地在哪里？"
            ]
        
        # 遍历查询并发送请求
        print(f"开始处理共 {len(df)} 条查询...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            query = row['query']
            result = send_query_request(api_url, query, sample_history)
            
            if result and result.get('success') and 'data' in result:
                query_type = result['data'].get('queryType')
                if query_type is not None:
                    df.at[idx, 'pred_label'] = query_type
                    df.at[idx, 'label_name'] = get_label_name(query_type)
            
            # 避免请求过于频繁
            time.sleep(0.1)
        
        return df
    except Exception as e:
        print(f"处理Excel文件时出错: {str(e)}")
        return None


def save_results(df, input_file):
    """保存处理结果到新的Excel文件"""
    try:
        # 生成输出文件名
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_results{file_ext}"
        
        # 保存到Excel
        df.to_excel(output_file, index=False)
        print(f"结果已保存到: {output_file}")
        
        # 输出统计信息
        print("\n分类结果统计:")
        stats = df['label_name'].value_counts()
        for label, count in stats.items():
            print(f"{label}: {count}条")
        
        return output_file
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
        return None


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="查询类型分类批量测试脚本")
    parser.add_argument("-f", "--file", required=True, help="包含查询的Excel文件路径")
    parser.add_argument("-u", "--url", default="http://10.19.16.193:8883/ai-test/yun/ai/text/query/rewrite", 
                        help="API请求URL")
    args = parser.parse_args()
    
    # 处理Excel文件
    result_df = process_excel(args.file, args.url)
    
    if result_df is not None:
        # 保存结果
        save_results(result_df, args.file)


if __name__ == "__main__":
    """
    python test_query_cls.py -f test_20250520.xlsx -u http://10.19.16.193:8883/ai-test/yun/ai/text/query/rewrite
    """
    main()
