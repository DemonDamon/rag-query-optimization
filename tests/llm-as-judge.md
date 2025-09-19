
# LLM-as-Judge评估方案设计

## 评估框架设计

### 1. 评判模型选择
| 模型 | 优势 | 适用场景 |
|------|------|---------|
| GPT-4o | 理解能力最强，评判最准确 | 高质量评估，结果作为黄金标准 |
| Claude-3.5-sonnet | 理解力强，中文表现好 | 高质量评估的替代选择 |
| Qwen2.5-72B | 国产大模型，成本低 | 大规模评估，预筛选 |

### 2. 评分维度与标准
```json
{
  "语义保持度": {
    "1分": "完全改变了原问题意图",
    "3分": "部分保留了原问题意图，但有重要信息丢失",
    "5分": "完全保留了原问题意图和关键信息"
  },
  "查询完整性": {
    "1分": "缺失多个关键要素，无法回答",
    "3分": "包含部分关键要素，可能影响回答质量", 
    "5分": "包含所有必要关键要素，可以完整回答"
  },
  "指代消解准确率": {
    "1分": "完全未解决指代问题或错误解读",
    "3分": "部分解决指代问题，有一定模糊性",
    "5分": "完全正确解决所有指代问题"
  }
}
```

### 3. 评估提示词模板
```
你是一位专业的问答系统质量评估专家。请评估以下查询改写的质量。

历史问题：
$HISTORY_QUERIES

当前问题：
$CURRENT_QUERY

改写后问题：
$REWRITTEN_QUERY

请从以下三个维度进行1-5分的评估（1分最低，5分最高）：

1. 语义保持度：改写后的查询是否保持了原始查询的意图和语义？
2. 查询完整性：改写后的查询是否包含了回答问题所需的所有关键信息？
3. 指代消解准确率：改写后的查询是否正确解决了指代问题（如代词、省略等）？

对于每个维度，请给出分数和简短理由。最后，请计算总分（三个维度的平均值）。
```

### 4. 评分流程实现
```python
def llm_judge_evaluation(history_queries, current_query, rewritten_query, judge_model="gpt-4"):
    # 构建提示词
    prompt = f"""
    你是一位专业的问答系统质量评估专家。请评估以下查询改写的质量。
    
    历史问题：
    {history_queries}
    
    当前问题：
    {current_query}
    
    改写后问题：
    {rewritten_query}
    
    请从以下三个维度进行1-5分的评估（1分最低，5分最高）：
    
    1. 语义保持度：改写后的查询是否保持了原始查询的意图和语义？
    2. 查询完整性：改写后的查询是否包含了回答问题所需的所有关键信息？
    3. 指代消解准确率：改写后的查询是否正确解决了指代问题（如代词、省略等）？
    
    对于每个维度，请给出分数和简短理由，按以下JSON格式输出：
    
    {
      "语义保持度": {"分数": X, "理由": "..."},
      "查询完整性": {"分数": X, "理由": "..."},
      "指代消解准确率": {"分数": X, "理由": "..."},
      "总分": X,
      "综合评价": "..."
    }
    """
    
    # 调用评判模型
    response = call_judge_model(prompt, model=judge_model)
    
    # 解析结果
    try:
        result = json.loads(response)
        return result
    except:
        # 备用解析方案，处理非严格JSON输出
        return extract_scores_from_text(response)
```

### 5. 批量评估与数据聚合
```python
def batch_evaluate(test_cases, rewrite_model, judge_model="gpt-4"):
    results = []
    
    for case in test_cases:
        # 使用改写模型生成结果
        rewritten_query = rewrite_model.generate(
            history_queries=case["history_queries"],
            current_query=case["current_query"]
        )
        
        # 使用评判模型评估
        scores = llm_judge_evaluation(
            case["history_queries"],
            case["current_query"],
            rewritten_query,
            judge_model
        )
        
        results.append({
            "case_id": case["id"],
            "original_query": case["current_query"],
            "rewritten_query": rewritten_query,
            "scores": scores
        })
    
    # 计算聚合指标
    aggregated_scores = {
        "语义保持度": sum(r["scores"]["语义保持度"]["分数"] for r in results) / len(results),
        "查询完整性": sum(r["scores"]["查询完整性"]["分数"] for r in results) / len(results),
        "指代消解准确率": sum(r["scores"]["指代消解准确率"]["分数"] for r in results) / len(results),
        "总分": sum(r["scores"]["总分"] for r in results) / len(results)
    }
    
    return results, aggregated_scores
```

### 6. 偏差控制与质量保证
- **多模型交叉验证**：使用2-3个不同大模型进行评估，取平均值
- **人工校准**：随机抽取10%样本进行人工评估，与模型评分比对校准
- **评分一致性检查**：计算模型间评分的肯德尔系数(Kendall's Tau)
- **重复评估**：对同一样本进行多次评估，检验稳定性

### 7. 结果呈现
- 定量指标：各维度平均分、标准差、中位数
- 样本分布：不同分数区间的样本分布直方图
- 典型案例：各分数段的代表性案例展示
- 对比分析：不同模型的评分对比雷达图