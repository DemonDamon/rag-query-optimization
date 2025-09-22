# 技术报告：解决小型LLM微调中的幻觉与上下文缺失问题


## 执行摘要

本技术报告为您在微调`qwen2.5-3b-instruct`模型进行查询重写任务时遇到的两大核心挑战——**模型幻觉**与**上下文缺失**——提供了深入的根本原因分析及一套系统的、可落地的解决方案。报告的核心论点是，通过结合先进的提示工程、模型蒸馏、参数高效微调（PEFT）技术以及遵循业界标准的API设计规范，可以显著提升小型语言模型的可靠性与对话能力。

**针对模型幻觉问题**，报告指出了其根源在于小型模型难以区分任务指令与少样本（Few-Shot）示例，导致“示例泄露”。解决方案双管齐下：
1.  **高级提示工程**：通过采用XML标签（如`<task>`, `<example>`）等结构化提示方法，为模型设定清晰的解析边界，有效隔离指令与示例内容[[2]](https://javier-marin.medium.com/what-is-xml-prompting-2e44cd8d5461)。同时，报告强调了优化少样本示例的相关性、数量（对小模型通常不超过10个）和顺序的重要性[[9]](https://arxiv.org/html/2509.13196v1)。
2.  **模型蒸馏与数据质量控制**：建议利用如GPT-4等强大的“教师模型”生成大规模、高质量的合成数据。关键在于结合思维链（Chain-of-Thought, CoT）提示词设计，并采纳如**FiNE框架**[[15]](https://aclanthology.org/2025.naacl-long.437.pdf)的理念，即使用教师模型自身作为“裁判”来对合成数据进行多维度打分、过滤与去重，从而确保微调数据的纯净度与多样性。

**针对上下文缺失问题**，即模型因未接收到完整的对话历史而无法理解指代关系，报告提出了两种解决方案：
1.  **理想方案（API层面修复）**：修改您的API接口，使其能够传递完整的、包含`user`和`assistant`角色的对话历史数组。后端服务必须负责将此标准化的`messages`数组动态转换为Qwen模型专用的“聊天模板”格式[[13]](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)，这是从根本上解决“有缺陷的对话历史”问题的关键[[1]](https://www.together.ai/blog/fine-tuning-llms-for-multi-turn-conversations-a-technical-deep-dive)。
2.  **变通方案（多级分类模型）**：在无法修改API的情况下，构建一个两阶段的分类流程。第一阶段沿用您已有的`<是否需要改写>`分类器。第二阶段，设计一个新的分类或提示任务，专门判断查询是否依赖于缺失的`assistant`回答。仅当查询不依赖于此部分上下文时，才执行重写。

最后，本报告整合上述策略，提出了一个包含数据生成、过滤、高效微调（推荐使用DoRA）及部署的端到端工作流程，并以表格形式清晰总结了各挑战对应的解决方案与关键实施细节，旨在为您提供一套全面、实用的技术指引。

## 1. 问题分析与诊断

您在使用`qwen2.5-3b-instruct`进行查询重写微调时遇到的问题非常典型，精准地反映了小型语言模型在实际应用中的两大核心痛点。

### 痛点一：模型幻觉与“示例泄露” (Example Leakage)

**现象**：模型在改写查询时，错误地将您在提示词（Prompt）中用作示例（Few-shot Example）的内容（如“Python和Java”）混入到最终的输出中。

**根本原因**：
小型语言模型的参数量有限，导致其在解析复杂提示时，区分“任务指令”与“上下文示例”的能力较弱。当指令与示例的边界模糊时，模型会错误地认为示例中的具体实体或短语也是任务要求的一部分，从而在生成时产生内容“泄漏”。这本质上是模型对上下文指令的错误解析所引发的一种幻奇[[3]](https://aiflowchat.com/blog/articles/how-xml-prompting-improves-your-ai-flows)。

### 痛点二：因上下文缺失导致的指代不明 (Defective Dialogue History)

**现象**：当用户的查询依赖于前一轮`assistant`的回答时（例如，在回答“Python和Java哪个更好？”后，用户追问“它的内存管理怎么样？”），模型无法正确改写，因为它只接收到了用户的历史查询，而没有接收到助手的历史回答。

**根本原因**：
这个问题的根源在于您的**API设计**以及相应的**微调数据构建方式**。您的接口只将用户的`queries`作为历史上下文，这构成了所谓的“有缺陷的对话历史”。模型在微调时，如果训练数据本身就不包含`assistant`的历史角色，那么它就永远学不会如何利用这部分信息来解析指代。它无法“看到”完整的对话流，因此在推理时自然也无法处理依赖于自身先前回答的查询[[1]](https://www.together.ai/blog/fine-tuning-llms-for-multi-turn-conversations-a-technical-deep-dive)。

接下来，我们将针对这两个核心问题，提供一套体系化的解决方案。

## 2. 解决方案：模型幻觉的应对策略

### 2.1 策略一：高级提示工程 (Prompt Engineering) - 立竿见影

这是最直接、成本最低的优化方式，核心思想是**为模型的输入信息建立清晰的结构边界**。

1.  **使用XML标签或结构化分隔符**：
    这是业界公认的最佳实践。不要将指令、示例和当前查询简单地用换行符连接。请使用XML风格的标签将不同部分包裹起来，为模型提供一种清晰的“提示语法”。

    **优化后的`<如何改写>`指令结构示例**：
    ```xml
    <task>
    你的任务是利用历史查询的上下文，解析并替换当前查询中的指代不明词语，使其更清晰、更具体。
    ...
    (此处省略其余指令)
    </task>
    
    <example>
    历史：
    Q1: Unity和Unreal Engine在性能上有什么区别？
    Q2: ...
    Q5: 哪个更适合初学者入门？
    当前查询: 它适合做啥类型游戏？
    改写结果: 它适合做啥类型游戏？
    </example>
    
    <example>
    历史：
    Q1: Python和Java哪个更高效？
    当前查询: 它的内存管理怎么样？
    改写结果: 它的内存管理怎么样？
    </example>
    
    <history>
    Q1: 唐宋元明清中那个朝代最短？
    </history>
    
    <current_query>
    哪个最长？
    </current_query>
    
    <rewritten_query>
    ```
    通过这种方式，模型能够更容易地学习到`<example>`标签内的内容仅供参考，其目标是填充`<rewritten_query>`，从而极大降低内容泄露的风险[[2]](https://javier-marin.medium.com/what-is-xml-prompting-2e44cd8d5461)。

2.  **优化少样本示例 (Few-Shot Examples)**：
    *   **相关性优先**：确保示例与真实任务场景高度相关。
    *   **“少即是多”**：研究表明，对小型模型而言，提供过多（例如超过10个）示例反而可能导致性能下降，即“少样本困境”[[9]](https://arxiv.org/html/2509.13196v1)。请尝试减少示例数量。
    *   **顺序敏感性**：示例的排列顺序对模型性能有巨大影响。建议通过实验调整示例顺序，找到最优组合。

### 2.2 策略二：模型蒸馏 (Model Distillation) - 釜底抽薪

如果提示工程优化后幻觉问题依然存在，根本解决方案是提升微调数据的质量与规模。模型蒸馏是实现这一目标的最高效手段。

1.  **教师-学生模型模式**：
    利用一个非常强大的“教师模型”（如GPT-4, Claude 3 Opus）来生成一个大规模、高质量的合成微调数据集，然后用这个数据集来训练您的`qwen2.5-3b-instruct`“学生模型”。

2.  **生成高质量合成数据**：
    *   **多样化的指令**：在向教师模型请求生成数据时，指令要多样化，覆盖各种边缘案例（例如，包含错别字、不规范表达、指代模糊的查询）。
    *   **融入思维链 (Chain-of-Thought, CoT)**：让教师模型在给出最终改写结果前，先输出它的推理过程。例如：“*分析：当前查询‘哪个最长’明显缺少主体，历史查询是关于‘唐宋元明清’这几个朝代。因此，‘哪个’指代的是这些朝代。改写：将主体补充完整。*” 将这些推理过程一同作为训练数据，可以教会学生模型如何“思考”，而不仅仅是模仿。

3.  **数据过滤与质量控制：应用FiNE框架理念**：
    教师模型也可能犯错。因此，一个严格的数据过滤流程至关重要。**FiNE框架**[[15]](https://aclanthology.org/2025.naacl-long.437.pdf)为此提供了极佳的思路：
    *   **使用教师模型作为“裁判”**：利用教师模型本身（或另一个强大的LLM）来为生成的每一个合成数据点打分。评分维度可包括：指令遵循度、逻辑准确性、事实正确性等。
    *   **多维度过滤**：基于上述评分，筛掉低质量、重复或有偏见的数据。
    *   **基于参考的修正**：对于一些有潜力但存在小瑕疵的数据，可以让“裁判”模型参考原始答案，生成一个更优的版本。

4.  **混合数据集策略**：
    公认的最佳实践是，使用**大规模、经过严格筛选的合成数据**作为训练主体，再辅以一小部分**高质量的人工标注“黄金”数据**进行最终校准。这可以确保模型在具备泛化能力的同时，其行为也与人类期望高度对齐。

## 3. 解决方案：上下文缺失的应对策略

### 3.1 理想方案：在API层面实现完整的对话历史传递

解决此问题的最根本、最彻底的方法是**修复数据链路**，确保模型在微调和推理时都能接收到完整的对话上下文。

1.  **修改API接口规范**：
    调整您的API，使其不再只传递`queries`字符串数组，而是传递一个结构化的`messages`对象数组。这已成为行业标准（例如OpenAI的Chat API）。每个对象都应包含`role`和`content`两个字段。

    **推荐的请求体结构 (JSON)**:
    ```json
    {
      "model": "qwen2.5-3b-instruct-finetuned",
      "messages": [
        {
          "role": "user",
          "content": "Python和Java哪个更高效？"
        },
        {
          "role": "assistant",
          "content": "这取决于具体的应用场景。在计算密集型任务中，Java通常因为其即时编译（JIT）而表现出更高的性能。而在快速开发和脚本任务中，Python则更具优势。"
        },
        {
          "role": "user",
          "content": "那它的内存管理怎么样？"
        }
      ]
    }
    ```

2.  **后端实现聊天模板 (Chat Template)**：
    您的API服务器后端在接收到上述`messages`数组后，**必须**将其转换为Qwen模型在训练时所能理解的特定字符串格式，即“聊天模板”。Qwen系列模型使用特殊的`<|im_start|>`和`<|im_end|>`等控制字符来包裹每个角色的发言[[13]](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)。

    **后端转换逻辑示例 (伪代码)**:
    ```python
    def convert_to_qwen_template(messages):
        prompt_string = ""
        for message in messages:
            role = message['role']
            content = message['content']
            prompt_string += f"<|im_start|>{role}
{content}<|im_end|>
"
        return prompt_string + "<|im_start|>assistant
" # 提示模型开始生成
    ```
    这个转换过程对前端调用者是透明的。

3.  **构建对应的微调数据**：
    您的微调数据集也必须遵循这种包含`user`和`assistant`角色的完整对话历史格式。每一个训练样本都应该是到当前轮次为止的完整对话流。只有这样，模型才能学会在上下文中理解指代关系。

### 3.2 变通方案：构建多级分类与条件重写流程

如果您暂时无法修改API，可以实施一个较为复杂的变通方案。

1.  **第一级：判断是否需要改写**
    *   继续使用您现有的`<是否需要改写>`分类器。它能处理大部分情况，如检测明确的代词、不规范用语等。

2.  **第二级：判断是否依赖`assistant`回答**
    *   如果第一级判断结果为`true`，则进入第二级。
    *   您需要额外训练一个**新的分类模型**或设计一个**新的提示任务**，其唯一目标是判断：“当前查询的模糊性是否源于对（缺失的）上一轮`assistant`回答的依赖？”。
    *   这个任务的判断依据是，当用户的提问（如“它怎么样？”、“详细说说”）本身高度抽象，且用户历史查询的最后一项是一个开放性问题或比较性问题时，大概率是依赖`assistant`的回答。

3.  **条件重写逻辑**：
    *   **IF** (第一级=`true`) **AND** (第二级=`false`):
        *   这表明查询需要改写，且改写的依据在`user`的历史查询中就能找到。此时，正常调用您的`<如何改写>`模型。
    *   **IF** (第一级=`true`) **AND** (第二级=`true`):
        *   这表明查询虽然需要改写，但关键信息在缺失的`assistant`回答中，无法安全改写。此时，**应放弃改写，直接返回原始查询**。您的`<如何改写>`指令中已经包含了这种“答案依赖场景不进行改写”的规则，这非常正确。

这个变通方案虽然复杂，但可以在不修改现有API的情况下，通过逻辑组合来规避错误的改写。

## 4. 集成工作流与最佳实践总结

```mermaid
graph TD
    A[需求分析: 定义Query改写任务] --> B(选择教师模型 e.g., GPT-4);
    B --> C{合成数据生成};
    C --> C1[1. 设计多样化CoT Prompt];
    C --> C2[2. 强调边缘案例生成];
    C --> D[生成大规模原始合成数据];
    D --> E(数据过滤与管理: FiNE框架);
    E --> E1[多因素初步过滤(教师模型作裁判)];
    E1 --> E2[基于参考的质量提升与修正];
    E2 --> E3[多样性分析与去重];
    E3 --> F[构建高质量混合微调数据集];
    F --> F1[添加少量人工黄金数据];
    F --> G(选择学生模型: qwen2.5-3b-instruct);
    G --> H{PEFT微调};
    H --> H1[选择方法: DoRA];
    H1 --> H2[训练轻量级适配器];
    H2 --> I[生成微调后的模型];
    I --> J(评估与迭代);
    J --> K[部署与API服务];

subgraph 高质量数据构建
    B; C; D; E; F; F1;
end

subgraph 高效模型适配
    G; H; I;
end
```

| 核心挑战 | 推荐解决方案 | 关键实施细节 | 引用/依据 |
| --- | --- | --- | --- |
| 模型幻觉，将示例内容泄露到输出中 | 结构化提示工程 | 使用XML标签（如`<task>`, `<example>`）严格区分指令和示例。优化示例数量（SLMs通常<10）和顺序。 | [[2]](https://javier-marin.medium.com/what-is-xml-prompting-2e44cd8d5461), [[9]](https://arxiv.org/html/2509.13196v1) |
| 微调数据质量不高，覆盖场景有限 | 模型蒸馏 + FiNE框架数据过滤 | 使用强大的教师模型，结合思维链（CoT）提示生成合成数据。应用FiNE理念，让教师模型作“裁判”进行多阶段过滤、修正和去重。 | [[15]](https://aclanthology.org/2025.naacl-long.437.pdf) |
| 多轮对话上下文追踪失败，无法处理对`assistant`回答的指代 | 修复API，遵循“聊天模板”构建微调数据 | API应传递包含`user`和`assistant`角色的完整`messages`历史数组。后端**必须**将此数组转换为模型特定的聊天模板字符串再输入模型。 | [[1]](https://www.together.ai/blog/fine-tuning-llms-for-multi-turn-conversations-a-technical-deep-dive), [[13]](https://huggingface.co/blog/qwen-3-chat-template-deep-dive) |
| 微调计算成本高昂，硬件资源受限 | 采用参数高效微调（PEFT） | 优先选择**DoRA**以在性能和效率之间取得最佳平衡。若显存是首要瓶颈，则选择QLoRA。为不同任务训练独立的轻量级适配器。 | [[18]](https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/) |

## 5. 结论

总而言之，您所遇到的挑战并非孤例，而是当前小型语言模型落地应用过程中的共性问题。解决这些问题需要系统性的思维，而非单一的技术“银弹”。

*   对于**模型幻觉**，其本质是模型对输入的“理解确定性”不足。我们必须从“喂给”模型的数据和指令的源头进行治理。通过**结构化提示**建立清晰的沟通语法，通过**模型蒸馏和FiNE式的数据过滤**提供高质量的精神食粮，是根治幻觉问题的正途。

*   对于**上下文缺失**，其本质是数据链路的“完整性”不足。最理想的解决方案是在**API架构层面进行标准化改造**，传递完整的对话历史，并由后端妥善处理特定于模型的聊天模板，这是确保模型能够进行连贯、有记忆的对话的技术基石。如果此路不通，设计**多级分类判断的变通逻辑**也能在很大程度上规避因信息不足而导致的灾难性改写错误。

通过上述策略的组合实施，您将能把`qwen2.sft-3b-instruct`微调成一个更可靠、更“聪明”的查询重写助手，充分发挥其在特定场景下的效能与效率优势。

## 6. 参考文献

1.  [https://www.together.ai/blog/fine-tuning-llms-for-multi-turn-conversations-a-technical-deep-dive](https://www.together.ai/blog/fine-tuning-llms-for-multi-turn-conversations-a-technical-deep-dive)
2.  [https://javier-marin.medium.com/what-is-xml-prompting-2e44cd8d5461](https://javier-marin.medium.com/what-is-xml-prompting-2e44cd8d5461)
3.  [https://aiflowchat.com/blog/articles/how-xml-prompting-improves-your-ai-flows](https://aiflowchat.com/blog/articles/how-xml-prompting-improves-your-ai-flows)
4.  [https://www.linkedin.com/posts/will-del-principe-57b18b2a5_nobody-talks-about-this-you-can-use-xml-activity-7356710713018490881-xxQl](https://www.linkedin.com/posts/will-del-principe-57b18b2a5_nobody-talks-about-this-you-can-use-xml-activity-7356710713018490881-xxQl)
5.  [https://www.sundeepteki.org/advice/the-definitive-guide-to-prompt-engineering-from-principles-to-production](https://www.sundeepteki.org/advice/the-definitive-guide-to-prompt-engineering-from-principles-to-production)
6.  [https://www.optimizesmart.com/prompting-text-markdown-json-schema-code-block/](https://www.optimizesmart.com/prompting-text-markdown-json-schema-code-block/)
7.  [https://www.linkedin.com/posts/imohitmayank_json-yaml-text-markdown-prompt-templates-activity-7267137544423657472-Tbr9](https://www.linkedin.com/posts/imohitmayank_json-yaml-text-markdown-prompt-templates-activity-7267137544423657472-Tbr9)
8.  [https://assets.amazon.science/8f/83/7407a5634a80a39e82b52ae935fe/on-mitigating-code-llm-hallucinations-with-api-documentation.pdf](https://assets.amazon.science/8f/83/7407a5634a80a39e82b52ae935fe/on-mitigating-code-llm-hallucinations-with-api-documentation.pdf)
9.  [https://arxiv.org/html/2509.13196v1](https://arxiv.org/html/2509.13196v1)
10. [https://arxiv.org/html/2509.13196v1](https://arxiv.org/html/2509.13196v1)
11. [https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/)
12. [https://www.promptingguide.ai/models/mistral-7b](https://www.promptingguide.ai/models/mistral-7b)
13. [https://huggingface.co/blog/qwen-3-chat-template-deep-dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
14. [https://aclanthology.org/2024.findings-acl.658.pdf](https://aclanthology.org/2024.findings-acl.658.pdf)
15. [https://aclanthology.org/2025.naacl-long.437.pdf](https://aclanthology.org/2025.naacl-long.437.pdf)
16. [https://www.redhat.com/en/topics/ai/lora-vs-qlora](https://www.redhat.com/en/topics/ai/lora-vs-qlora)
17. [https://pravi.tech/posts/fine-tuning/](https://pravi.tech/posts/fine-tuning/)
18. [https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/](https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/)
19. [https://unfoldai.com/catastrophic-forgetting-llms/](https://unfoldai.com/catastrophic-forgetting-llms/)