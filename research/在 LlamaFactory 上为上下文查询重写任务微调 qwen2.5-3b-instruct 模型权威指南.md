# 在 LlamaFactory 上为上下文查询重写任务微调 qwen2.5-3b-instruct 模型权威指南


上下文查询重写（Contextual Query Rewriting, CQR）是现代对话式人工智能系统的核心组成部分。其主要功能是利用对话历史，将依赖于上下文的、模糊的用户查询转化为独立的、明确的问题 [[1]](https://aclanthology.org/2023.emnlp-industry.41.pdf)。这一过程对于提高检索增强生成（RAG）系统的准确性、对话状态跟踪的精确度以及保障整体对话的连贯性至关重要。然而，在处理长对话历史时，特别是在微调计算资源要求较低的小型语言模型时，一个巨大的挑战随之而来。这些模型固有的上下文窗口限制，使得在没有有效上下文管理策略的情况下处理大量对话变得异常困难 [[2]](https://arxiv.org/html/2410.02660v1)。

本指南旨在为您提供一个全面、端到端的实践流程，指导您如何在LlamaFactory生态系统中使用`qwen2.5-3b-instruct`这一小型语言模型，训练出能够执行高质量上下文查询重写的模型。LlamaFactory是一个统一且高效的大型语言模型微调框架 [[3]](https://github.com/hiyouga/LLaMA-Factory)。本指南将详尽覆盖从数据准备、长上下文压缩，到模型训练（包括监督微调SFT和基于直接偏好优化的强化学习DPO），直至最终高级评估的全过程。我们将提供可直接操作的配置方案与代码片段，为技术人员和开发者提供一份坚实的基础实践手册。

## 核心问题解析：如何通过SFT+RL实现长上下文压缩与高效重写？

在深入技术细节之前，我们首先直接回应您的核心问题：“如果我要结合上下文一起给到qwen2.5-3b-instruct，在llamafactory上，如何实现sft+rl的训练实现对长上下文的压缩，并且给出改写高预期的效果？”

首先，需要明确这里的“压缩”并非指传统意义上的数据压缩（如zip），而是**语义层面的信息蒸馏**。其核心目标是训练模型从冗长、可能包含干扰信息的对话历史中，智能地识别、提取并聚焦于与当前用户查询最相关的关键信息，最终生成一个简洁、完整且无歧义的重写查询。

为了实现这一目标，我们推荐采用一种“**SFT基础训练 + DPO偏好对齐**”的两阶段训练策略。这两种方法的协同作用是达成高质量重写效果的关键：

1.  **第一阶段：监督微调（SFT） - 教授基础技能**
    *   **目标**：SFT是模型学习任务基础的第一步。在这一阶段，我们通过大量“输入-输出”样本对，教会模型理解查询重写任务的基本范式。模型学习如何根据给定的“对话历史 + 当前查询”（输入），生成一个语法正确、与输入内容相关的重写查询（输出）。
    *   **作用**：此阶段为模型奠定了一个坚实的性能基础。完成SFT后，模型已经具备了执行查询重写的核心能力，但其输出可能仍有优化空间，例如可能不够精炼，或者在面对极其复杂的上下文时无法做出最佳判断。

2.  **第二阶段：直接偏好优化（DPO） - 精炼判断能力**
    *   **目标**：DPO属于强化学习（RL）的范畴，其目的是在SFT的基础上进一步对齐模型的“价值观”，使其输出更符合人类的偏好。与需要训练独立奖励模型的传统RL方法（如PPO）相比，DPO通过“更优/更差”的偏好数据对直接进行优化，过程更简单、训练更稳定 [[5]](https://www.labellerr.com/blog/dpo-vs-ppo-for-llm-all/)。
    *   **作用**：在DPO阶段，我们向模型展示对于同一个输入，哪一个重写版本是“更好”的（chosen），哪一个是“较差”的（rejected）。通过这种方式，模型不仅知道*如何*重写，更学会了*如何更好地*重写。它学习辨别并生成更连贯、上下文理解更准确、信息保留更完整且语言更精炼的查询。这正是实现“改写高预期效果”的关键一步。

总结而言，SFT为模型搭建了骨架，教会了它“走路”，而DPO则精雕细琢，教会了它如何“优雅地舞蹈”。本指南接下来将为您详细拆解如何一步步实现这个从“会”到“精”的完整训练流程。

```mermaid
graph TD
    A[开始: 原始对话数据] --> B{步骤一: 数据准备};
    B --> C[长上下文压缩<br>(滑动窗口策略)];
    C --> D{格式化数据集};
    D --> E[SFT 数据集 (.json)];
    D --> F[DPO 偏好数据集 (.json)];

    subgraph 模型训练流程
        G[阶段一: 监督微调<br>(使用 SFT 数据集)];
        H[阶段二: 直接偏好优化<br>(使用 DPO 偏好数据集)];
        G --> H;
    end

    E --> G;
    F --> H;

    H --> I{步骤三: 模型评估};
    I --> J[高级评估指标<br>(BERTScore, 困惑度等)];
    J --> K[完成: 微调后的<br>查询重写模型];
```

## 步骤一：数据准备

高质量的数据是模型成功的基石。此阶段的核心任务是处理长对话历史并构建符合LlamaFactory要求的SFT和DPO数据集。

### 1.1. 长上下文压缩：滑动窗口（Sliding Window）策略

鉴于小型模型的上下文窗口限制，必须对长对话历史进行预处理。**滑动窗口**策略因其实现简单且效果显著而被广泛推荐。该方法仅保留最近的`k`轮对话，因为这些对话通常与当前查询的关联性最强，从而在保留关键上下文的同时，有效缩减了输入长度。

*   **实现方式**: 在将数据送入LlamaFactory之前，通过一个独立的Python脚本来执行。该脚本负责读取原始对话数据，并根据设定的窗口大小`k`来截取最近的对话轮次。
*   **计算成本**: 极低，主要涉及文本的计数与截断操作。

**滑动窗口转换示例：**
假设原始对话有10轮，我们设定滑动窗口大小`k=3`。

*   **原始长历史**: 包含从第1轮到第10轮的完整对话。
*   **滑动窗口处理后**: 脚本仅抽取出最近的3轮对话，即第8、9、10轮。
*   **最终输入LlamaFactory的`input`字段格式**:
    ```
    "input": "对话历史: [第8轮用户], [第8轮助手], [第9轮用户], [第9轮助手], [第10轮用户]
当前查询: [用户的最后一个问题]"
    ```

### 1.2. 构建SFT数据集

在第一阶段的监督微调（SFT）中，数据需要遵循LlamaFactory支持的Alpaca风格JSON格式。我们将处理过的对话历史与当前查询合并到`input`字段中。

**SFT数据集JSON示例:**
```json
[
  {
    "instruction": "请根据提供的对话历史，将当前用户查询改写成一个独立的、信息完整的句子。",
    "input": "对话历史: 用户: 今天天气怎么样? 助手: 今天是晴天，25摄氏度。 用户: 那明天呢?
当前查询: 明天天气",
    "output": "请问明天的天气怎么样？"
  }
]
```

### 1.3. 构建DPO偏好数据集

在第二阶段的直接偏好优化（DPO）中，我们需要一个包含偏好对的数据集。每条数据都应包含一个“更优”（`chosen`）的重写结果和一个“较差”（`rejected`）的重写结果。

**DPO数据集JSON示例:**
```json
[
  {
    "instruction": "请根据提供的对话历史，将当前用户查询改写成一个独立的、信息完整的句子。",
    "input": "对话历史: 用户: 今天天气怎么样? 助手: 今天是晴天，25摄氏度。 用户: 那明天呢?
当前查询: 明天天气",
    "chosen": "请问明天的天气怎么样？",
    "rejected": "明天的天气？"
  }
]
```

**如何生成高质量的偏好数据？**

构建有效的DPO数据集是整个流程的重中之重。以下是几种实用的策略：

1.  **使用强大的“裁判”大模型生成**: 利用能力更强的模型（如GPT-4、Claude 3 Opus）作为“裁判”，为给定的`input`生成多个候选重写版本，并让其评选出最优（作为`chosen`）和次优（作为`rejected`）的答案。这是一种自动化且高效的生成方式 [[8]](https://medium.com/@wonseok.chris.choi/generating-synthetic-datasets-for-direct-preference-optimization-fine-tuning-of-llms-slms-f8c6c9fd7e07)。

2.  **程序化生成“较差”样本**:
    *   **信息截断**: 从一个高质量的重写版本（可作为`chosen`）中，故意删除关键实体或信息，生成一个信息不完整的`rejected`版本。
    *   **引入模糊性**: 生成一个比原始查询略好，但仍然不够明确的`rejected`版本。
    *   **风格变换**: 生成一个在语法或风格上存在瑕疵（例如，过于口语化或生硬）的`rejected`版本 [[9]](https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data)。

3.  **复用SFT数据**: SFT数据集本身就是`chosen`样本的绝佳来源。你可以使用SFT阶段训练好的模型，对SFT数据集中的`input`进行多次推理（例如，使用不同的`temperature`参数），生成多个候选答案。然后，可以通过启发式规则（如长度、关键词覆盖率）或另一个评分模型，从中挑选出质量稍逊的样本作为`rejected`。

## 步骤二：模型训练 - 两阶段流程

我们的模型训练流程分为两个核心阶段：首先是监督微调（SFT），然后是直接偏好优化（DPO）。

### 2.1. 阶段一：监督微调 (SFT)

此阶段的目标是让模型掌握查询重写的基础能力。我们使用SFT数据集，通过LoRA（Low-Rank Adaptation）这种高效微调技术，对`qwen2.5-3b-instruct`模型进行训练。

**`qwen2.5-3b-instruct` SFT LoRA 训练配置示例:**

以下命令展示了如何使用`torchrun`在4个GPU上启动SFT训练。请根据您的实际环境修改设备号和文件路径。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 torchrun --nproc_per_node 4 src/train.py \
    --model_name_or_path unsloth/Qwen2.5-3B-Instruct \
    --do_train \
    --dataset your_sft_dataset_name \
    --finetuning_type lora \
    --lora_target all \
    --output_dir output/qwen2.5-3b-sft \
    --template qwen2_5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```
*   `--model_name_or_path`: 指定基础模型，这里我们使用unsloth优化过的Qwen2.5-3B模型以提升效率 [[4]](https://qwen.readthedocs.io/en/v2.5/training/SFT/llama_factory.html)。
*   `--dataset`: 指定您准备好的SFT数据集的名称。
*   `--finetuning_type lora`: 采用LoRA进行高效参数微调。
*   `--lora_target all`: 将LoRA应用到模型的所有线性层，以获得更好的效果。
*   `--output_dir`: 指定模型检查点和日志的保存路径。这个路径在DPO阶段会作为输入。
*   `--template qwen2_5`: 指定使用Qwen2.5模型的对话模板。
*   `--fp16`: 启用混合精度训练以加速计算并节省显存。

### 2.2. 阶段二：直接偏好优化 (DPO)

在SFT模型具备了基础能力后，我们使用DPO进一步提升其输出质量，使其更符合人类偏好。在LlamaFactory中，DPO被证明是比PPO更稳定且易于实现的强化学习对齐方法 [[5]](https://www.labellerr.com/blog/dpo-vs-ppo-for-llm-all/)。

**DPO 训练配置示例:**

DPO的训练命令与SFT类似，但需要调整`stage`参数，并将`model_name_or_path`指向SFT阶段的输出目录。

```bash
llamafactory-cli train \
    --stage dpo \
    --model_name_or_path output/qwen2.5-3b-sft \
    --do_train \
    --dataset your_preference_dataset_name \
    --template qwen2_5 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir output/qwen2.5-3b-dpo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --learning_rate 5e-7 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```
*   `--stage dpo`: 明确指定当前是DPO训练阶段。
*   `--model_name_or_path output/qwen2.5-3b-sft`: **关键步骤！** 这里加载的是SFT阶段训练好的模型权重，我们在此基础上进行偏好对齐。
*   `--dataset your_preference_dataset_name`: 指定包含`chosen`和`rejected`字段的DPO偏好数据集。
*   `--learning_rate 5e-7`: DPO阶段通常需要使用比SFT更小的学习率，以进行更精细的调整。
*   `--num_train_epochs 1.0`: DPO训练通常不需要像SFT那样进行多轮，1个epoch往往足够。

## 步骤三：评估

对于查询重写这类生成式任务，简单的词汇重叠度指标（如BLEU、ROUGE）往往无法准确评估模型性能。我们需要一套更关注语义、流畅度和信息完整性的高级评估指标。

| 指标类别 | 具体指标 | 描述 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| 语义相似度 | BERTScore | 使用BERT等预训练模型的词嵌入来计算生成文本和参考文本之间的相似度，能够捕捉深层次的语义信息 [[6]](https://arxiv.org/abs/1904.09675)。 | 能很好地捕捉同义词和释义，与人类判断的相关性高。 | 计算量相对较大，评估速度较慢。 |
| 语义相似度 | 句子嵌入余弦相似度 | 使用Sentence-BERT等模型将整个重写查询和参考查询转换为向量，然后计算它们之间的余弦相似度。 | 计算效率高，能有效评估句子层面的整体语义一致性。 | 评估效果高度依赖于所选句子嵌入模型的质量。 |
| 用户满意度代理指标 | 困惑度 (Perplexity) | 衡量语言模型对一个句子的“惊讶程度”。困惑度越低，说明模型认为这个句子越流畅、越合乎语法 [[7]](https://www.cfilt.iitb.ac.in/resources/surveys/2025/harsh_LLM4Eval_survey_paper%20(1).pdf)。 | 可以自动化地衡量输出的流畅度和语法质量。 | 无法评估语义的准确性或与上下文的相关性。一个流畅但错误的重写也可能有低困惑度。 |
| 用户满意度代理指标 | 关键实体保留率 | 检查对话历史和原始查询中的关键实体（如人名、地名、产品名）是否在重写查询中被准确地保留或整合。 | 直接衡量了信息的完整性和一致性，有助于发现信息遗漏问题。 | 需要一个强大的命名实体识别（NER）系统，且该系统可能需要针对特定领域进行微调。 |
| 清晰与简洁度 | 可读性分数 | 使用如Flesch-Kincaid等算法来评估文本的理解难度。分数越好通常代表查询越清晰易懂。 | 易于计算，为查询的清晰度提供了一个量化参考。 | 主要为人类阅读设计，与机器查询系统的处理效率不完全相关。 |

## 结论

本指南详细阐述了一个在LlamaFactory框架内，使用`qwen2.5-3b-instruct`模型进行上下文查询重写任务微调的端到端流程。我们推荐的 workflow 强调了实用且高效的策略，总结如下：

1.  **数据预处理**: 采用**滑动窗口**策略有效管理长对话历史，避免超出模型上下文窗口限制。
2.  **两阶段训练**:
    *   首先通过**监督微调（SFT）**，让模型掌握查询重写的核心技能。
    *   然后通过**直接偏好优化（DPO）**，精细调整模型输出，使其更符合人类对高质量、高连贯性重写的偏好。
3.  **综合评估**: 采用一套包含语义相似度、用户满意度代理指标和清晰度的**高级评估矩阵**，以全面、准确地衡量模型性能。

通过遵循从精心的数据准备，到系统的两阶段训练，再到全面的评估这一结构化路径，开发者可以高效地训练出小而精的查询重写模型。这类模型能够生成高质量、充分理解上下文的查询，从而显著提升对话式AI系统（尤其是RAG应用）的整体性能和用户体验。

## 参考文献
[1] Liu, Z., et al. (2023). *Improving Contextual Query Rewrite for Conversational AI Agents through User Preference Feedback Learning*. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track. https://aclanthology.org/2023.emnlp-industry.41.pdf

[2] Li, S., et al. (2024). *How to Train Long-Context Language Models (Effectively)*. arXiv preprint arXiv:2410.02660. https://arxiv.org/html/2410.02660v1

[3] Zheng, Z., et al. (2024). *LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ Language Models*. arXiv preprint arXiv:2403.13372. https://github.com/hiyouga/LLaMA-Factory

[4] Qwen Team. (2024). *Supervised Finetuning - LlamaFactory*. Qwen Documentation. https://qwen.readthedocs.io/en/v2.5/training/SFT/llama_factory.html

[5] Labellerr. (2024). *DPO vs PPO: How To Align LLM*. Labellerr Blog. https://www.labellerr.com/blog/dpo-vs-ppo-for-llm-all/

[6] Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2019). *BERTScore: Evaluating Text Generation with BERT*. arXiv preprint arXiv:1904.09675. https://arxiv.org/abs/1904.09675

[7] Kashid, H., & Bhattacharyya, P. (2025). *Evaluating Large Language Models: A Survey*. CFILT, IIT Bombay. https://www.cfilt.iitb.ac.in/resources/surveys/2025/harsh_LLM4Eval_survey_paper%20(1).pdf

[8] Choi, W. (2024). *Generating Synthetic Datasets for Direct Preference Optimization (DPO) Fine-tuning of LLMs/SLMs*. Medium. https://medium.com/@wonseok.chris.choi/generating-synthetic-datasets-for-direct-preference-optimization-fine-tuning-of-llms-slms-f8c6c9fd7e07

[9] Anyscale. (2023). *Direct Preference Optimization with Synthetic Data*. Anyscale Blog. https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data