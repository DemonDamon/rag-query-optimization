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
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

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
        if 'query' not in df.columns or 'labels' not in df.columns:
            print("错误: Excel文件必须包含 'query' 和 'labels' 两列")
            return None
        
        # 添加新列
        df['pred_label'] = None
        df['label_name'] = None
        df['pred_label_name'] = None # 用于存储预测的中文标签名
        
        # 将labels列转为int类型，无效值填充为-1或其他标记，便于后续过滤
        df['labels'] = pd.to_numeric(df['labels'], errors='coerce').fillna(-1).astype(int)
        
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
                    df.at[idx, 'pred_label_name'] = get_label_name(query_type)
                
                # 填充真实标签的中文名
                actual_label_int = row['labels']
                if actual_label_int != -1: # 过滤掉转换失败的标签
                    df.at[idx, 'label_name'] = get_label_name(actual_label_int)
            
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
        # 统计真实标签分布
        print("真实标签分布:")
        actual_stats = df['label_name'].value_counts()
        for label, count in actual_stats.items():
            print(f"  {label}: {count}条")
        
        # 统计预测标签分布
        print("\n预测标签分布:")
        pred_stats = df['pred_label_name'].value_counts()
        for label, count in pred_stats.items():
            print(f"  {label}: {count}条")
        
        # 计算并打印各项指标
        # 过滤掉真实标签或预测标签无效的行
        valid_df = df[(df['labels'] != -1) & (df['pred_label'].notna())].copy()
        valid_df['pred_label'] = valid_df['pred_label'].astype(int)
        
        if not valid_df.empty:
            y_true = valid_df['labels']
            y_pred = valid_df['pred_label']
            
            target_names = [get_label_name(label) for label in sorted(list(set(y_true) | set(y_pred)))]
            # 如果target_names为空（例如所有label都是-1或预测为空），classification_report会报错
            if not target_names:
                print("\n没有有效的标签用于计算指标。")
                return output_file
            
            print("\n性能指标:")
            print(f"整体准确率 (Accuracy): {accuracy_score(y_true, y_pred):.4f}\n")
            
            # classification_report 要求labels参数与target_names对应
            unique_labels = sorted(list(set(y_true) | set(y_pred)))
            report = classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, zero_division=0)
            print(report)
        else:
            print("\n没有有效的预测结果用于计算性能指标。")
        
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
    python test_query_cls.py -f ..\data\query分类微调数据_已标注_20250520_201226.xlsx -u http://10.19.16.193:8883/ai-test/yun/ai/text/query/rewrite
    """
    main()
