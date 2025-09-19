#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import yaml
import json
import argparse
from tqdm import tqdm

"""
CLI 示例:
# 生成查询分类（query classification）训练数据
python lf_data_builder_for_querycls.py --excel query分类微调数据_已标注_20250520_201226.xlsx --prompt ../src/classify_prompts.yaml --output ./querycls/querycls_train.json --mode querycls
"""

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

def format_history_queries(row):
    """
    格式化历史查询，跳过空值。
    如果没有历史查询，返回空字符串而不是None。
    """
    history = []
    for i in range(1, 6):  # Q1到Q5
        col_name = f'Q{i}'
        if col_name in row and pd.notna(row[col_name]) and row[col_name]:
            history.append(f"Q{i}: {row[col_name]}")
    return "\n".join(history)

def build_judge_data(df, prompt_template):
    """
    构建判断模型的训练数据
    """
    training_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建判断模型数据"):
        # 跳过缺少当前查询的行
        if pd.isna(row['CurrentQuery']) or not row['CurrentQuery']:
            continue
            
        # 获取历史查询（即使没有历史查询也继续处理）
        history_text = format_history_queries(row)
        
        # 构建完整提示词
        prompt = prompt_template.format(
            history_queries=history_text,
            current_query=row['CurrentQuery']
        )
        
        # 确保NeedRewrite不为空
        if pd.isna(row['NeedRewrite']):
            continue
            
        # 获取标注结果
        result = "true" if row['NeedRewrite'] == True else "false"
        
        # 添加到训练数据
        training_data.append({
            "instruction": prompt,
            "input": "",
            "output": result
        })
    
    return training_data

def build_rewrite_data(df, prompt_template):
    """
    构建改写模型的训练数据
    """
    training_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建改写模型数据"):
        # 跳过不需要改写或缺少当前查询或预期查询的行
        if not row['NeedRewrite'] or pd.isna(row['CurrentQuery']) or pd.isna(row['ExpectedQuery']):
            continue
            
        # 获取历史查询（即使没有历史查询也继续处理）
        history_text = format_history_queries(row)
        
        # 构建完整提示词
        prompt = prompt_template.format(
            history_text=history_text,
            current_query=row['CurrentQuery']
        )
        
        # 添加到训练数据
        training_data.append({
            "instruction": prompt,
            "input": "",
            "output": row['ExpectedQuery']
        })
    
    return training_data

def build_querycls_data(df, prompt_template):
    """
    根据包含"query"和"labels"字段的 Excel 标注数据，构建查询分类模型训练数据。

    生成格式示例：
    {
        "instruction": "<拼接后的提示词>",
        "input": "",
        "output": "{\"queryType\": 2}"
    }
    """

    if 'query' not in df.columns or 'labels' not in df.columns:
        print("错误：Excel 中必须包含 'query' 和 'labels' 两列")
        return []

    training_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建查询分类数据"):
        query_text = row.get('query')
        label_val = row.get('labels')

        # 基本校验
        if pd.isna(query_text) or query_text == "":
            continue

        if pd.isna(label_val):
            continue

        # 尝试转换为 int，过滤异常值
        try:
            label_int = int(label_val)
        except ValueError:
            print(f"跳过行：无法将 labels 转为整数 -> {label_val}")
            continue

        # 生成 prompt。部分模板可能仅包含 {current_query} 占位符
        try:
            prompt = prompt_template.format(current_query=query_text)
        except KeyError:
            # 若模板尚包含 history_queries 等占位符，为其传入空字符串
            prompt = prompt_template.format(current_query=query_text, history_queries="")

        # 构造输出，保持与提示词一致的 JSON 结构（字符串形式）
        output_str = json.dumps({"queryType": label_int}, ensure_ascii=False)

        training_data.append({
            "instruction": prompt,
            "input": "",
            "output": output_str
        })

    return training_data

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将Excel数据转换为LlamaFactory训练所需的JSON格式')
    parser.add_argument('--excel', required=True, help='输入的Excel文件路径')
    parser.add_argument('--prompt', required=True, help='提示词YAML文件路径')
    parser.add_argument('--output', required=True, help='输出的JSON文件路径')
    parser.add_argument('--mode', required=True, choices=['judge', 'rewrite', 'querycls'], help='构建模式: judge(判断) / rewrite(改写) / querycls(分类)')
    parser.add_argument('--version', default='v1', help='使用的提示词版本，默认为v1')
    
    args = parser.parse_args()
    
    # 确保输入文件存在
    if not os.path.exists(args.excel):
        print(f"错误：Excel文件不存在: {args.excel}")
        return
    
    if not os.path.exists(args.prompt):
        print(f"错误：提示词文件不存在: {args.prompt}")
        return
    
    # 读取Excel文件
    try:
        df = pd.read_excel(args.excel)
        print(f"成功读取Excel文件，包含 {len(df)} 行数据")
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return
    
    # 加载提示词模板
    prompt_template = load_yaml_prompt(args.prompt, args.version)
    
    if not prompt_template:
        print("加载提示词模板失败，请检查文件路径和内容")
        return
    
    # 根据模式构建训练数据
    training_data = []
    
    if args.mode == 'judge':
        training_data = build_judge_data(df, prompt_template)
        print(f"构建判断模型训练数据完成，共 {len(training_data)} 条")
    elif args.mode == 'rewrite':
        training_data = build_rewrite_data(df, prompt_template)
        print(f"构建改写模型训练数据完成，共 {len(training_data)} 条")
    elif args.mode == 'querycls':
        training_data = build_querycls_data(df, prompt_template)
        print(f"构建查询分类训练数据完成，共 {len(training_data)} 条")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 保存训练数据
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练数据已保存至 {args.output}")
    print("处理完成！")

if __name__ == "__main__":
    main()
