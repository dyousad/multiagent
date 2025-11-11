# HotpotQA Multi-Agent Evaluation Guide

本文档说明如何使用扩展后的多智能体协作框架在 HotpotQA 数据集上进行评估。

## 新增功能

### 1. HotpotQA 环境支持 (Phase 1)
- **文件**: `src/environment_hotpotqa.py`
- **功能**: 加载和处理 HotpotQA 数据集,支持多跳问答评估
- **特性**:
  - 自动计算 Exact Match 和 F1 分数
  - 处理 HotpotQA 特定的数据格式
  - 支持不同的数据分割 (validation/train/test)

### 2. 问题分解智能体 (Phase 2)
- **文件**: `src/decomposer_agent.py`
- **功能**: 将复杂的多跳问题分解为简单的子问题
- **特性**:
  - 使用 LLM 自动识别推理步骤
  - 生成有序的子问题列表
  - 与主智能体系统集成

### 3. 动态信用分配 (Phase 3)
- **文件**: `src/reward_manager.py` (更新)
- **功能**: 基于反事实分析的测试时信用重分配
- **方法**:
  - `update_credits_dynamic()`: 计算每个智能体的边际贡献
  - `get_credit_entropy()`: 计算信用分布的熵
  - `reset_credits()`: 重置信用分数

### 4. 评估指标 (Phase 4)
- **文件**: `src/evaluation.py`
- **指标**:
  - 多跳问答准确率 (Exact Match)
  - Token-level F1 分数
  - Gini 公平性指数
  - Jain 公平性指数
  - Shannon 熵

### 5. 可视化工具
- **文件**: `scripts/plot_hotpotqa_results.py`
- **图表**:
  - 静态 vs 动态信用熵对比
  - 每个样本的性能指标
  - 智能体间的信用分布
  - 聚合指标摘要

## 使用方法

### 快速开始

#### 1. 运行 HotpotQA 评估

```bash
# 使用默认环境运行单个 HotpotQA 样本
python src/main.py --env hotpotqa --task 0 --num_agents 3

# 指定数据路径和样本数量
python src/main.py \
  --env hotpotqa \
  --data_path data/hotpot_dev_fullwiki_v1.json \
  --max_samples 100 \
  --num_agents 3 \
  --reward shapley
```

#### 2. 运行完整的 HotpotQA 实验

```bash
# 运行实验脚本(包含问题分解和动态信用分配)
python scripts/run_hotpotqa_experiments.py
```

该脚本会:
- 在 10 个 HotpotQA 样本上评估
- 使用问题分解智能体
- 计算静态和动态信用分配
- 保存详细结果到 `results/hotpotqa/hotpotqa_results.json`

#### 3. 生成可视化结果

```bash
# 绘制所有可视化图表
python scripts/plot_hotpotqa_results.py \
  --results results/hotpotqa/hotpotqa_results.json \
  --output_dir results/hotpotqa/plots
```

生成的图表:
- `credit_entropy_comparison.png`: 静态 vs 动态熵对比
- `performance_metrics.png`: EM 和 F1 分数
- `credit_distribution.png`: 智能体信用分布
- `aggregate_summary.png`: 聚合指标摘要
- `entropy_histogram.png`: 熵分布直方图

### 高级配置

#### 自定义实验参数

编辑 `scripts/run_hotpotqa_experiments.py`:

```python
results = run_hotpotqa_experiment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=50,  # 增加样本数量
    num_agents=4,     # 增加智能体数量
    model_identifier="deepseek-ai/DeepSeek-V3",
    use_decomposer=True,      # 启用问题分解
    use_dynamic_credit=True,  # 启用动态信用
    output_dir=Path("results/hotpotqa")
)
```

#### 配置文件设置

在 `config.yaml` 中配置 HotpotQA 环境:

```yaml
environment:
  type: "hotpotqa"
  max_rounds: 5
  mode: "sequential"

  hotpotqa:
    data_path: "data/hotpot_dev_fullwiki_v1.json"
    max_samples: 100
    split: "validation"
```

## 实验输出

### 结果文件结构

```json
{
  "config": {
    "data_path": "...",
    "max_samples": 10,
    "num_agents": 3,
    "use_decomposer": true,
    "use_dynamic_credit": true
  },
  "samples": [
    {
      "sample_id": 0,
      "question": "...",
      "ground_truth": "...",
      "predicted_answer": "...",
      "exact_match": true,
      "f1_score": 0.95,
      "static_rewards": {"agent_0": 0.33, ...},
      "dynamic_credits": {"agent_0": 0.42, ...},
      "static_entropy": 1.09,
      "dynamic_entropy": 1.15
    }
  ],
  "aggregate": {
    "exact_match_accuracy": 0.70,
    "average_f1": 0.82,
    "average_static_entropy": 1.10,
    "average_dynamic_entropy": 1.18
  }
}
```

### 关键指标说明

- **Exact Match (EM)**: 预测答案与真实答案完全匹配的比例
- **F1 Score**: Token 级别的精确率和召回率的调和平均
- **Static Entropy**: 基于 Shapley 值的信用分布熵 (较低表示信用集中)
- **Dynamic Entropy**: 基于反事实分析的信用分布熵 (较高通常表示更公平)
- **Entropy Improvement**: 动态熵 - 静态熵 (正值表示公平性提升)

## 核心组件使用示例

### DecomposerAgent

```python
from decomposer_agent import DecomposerAgent

# 创建分解器
decomposer = DecomposerAgent(
    agent_id="decomposer",
    model_identifier="deepseek-ai/DeepSeek-V3"
)

# 分解问题
result = decomposer.decompose_question(
    question="Which movie has a higher budget, Inception or Avatar?",
    context="..."
)

print(result['sub_questions'])
# ['What is the budget of Inception?',
#  'What is the budget of Avatar?',
#  'Which budget is higher?']
```

### 动态信用分配

```python
from reward_manager import RewardManager

reward_manager = RewardManager(base_reward=1.0)

# 动态更新信用
credits = reward_manager.update_credits_dynamic(
    agent_outputs={"agent_0": "...", "agent_1": "..."},
    final_answer="predicted answer",
    ground_truth="true answer",
    evaluate_fn=custom_eval_function  # 可选
)

# 计算熵
entropy = reward_manager.get_credit_entropy(credits)
print(f"Credit Entropy: {entropy:.3f}")
```

### 评估指标

```python
from evaluation import calculate_all_metrics, print_metrics_report

metrics = calculate_all_metrics(
    predictions=["answer1", "answer2"],
    gold_answers=["truth1", "truth2"],
    static_credits=[{"agent_0": 0.5, "agent_1": 0.5}],
    dynamic_credits=[{"agent_0": 0.6, "agent_1": 0.4}]
)

print_metrics_report(metrics)
```

## 预期结果

成功运行后,你应该看到:

1. **控制台输出**:
   - 每个样本的问题、预测、EM、F1
   - 动态信用分数
   - 聚合指标摘要

2. **结果文件**:
   - `results/hotpotqa/hotpotqa_results.json`

3. **可视化图表**:
   - 5个 PNG 图表在 `results/hotpotqa/plots/`

## 文件清单

### 新增/修改的文件

```
src/
├── environment_hotpotqa.py      # HotpotQA 环境 (新增)
├── decomposer_agent.py          # 问题分解智能体 (新增)
├── evaluation.py                # 评估指标模块 (新增)
├── reward_manager.py            # 动态信用分配 (更新)
├── controller.py                # 支持分解器集成 (更新)
└── main.py                      # 支持 HotpotQA 环境 (更新)

scripts/
├── run_hotpotqa_experiments.py  # HotpotQA 实验脚本 (新增)
└── plot_hotpotqa_results.py     # HotpotQA 可视化脚本 (新增)

config.yaml                      # HotpotQA 配置 (更新)
HOTPOTQA_GUIDE.md               # 本指南 (新增)
```

## 故障排查

### 常见问题

1. **数据文件未找到**
   ```
   FileNotFoundError: Data file not found: data/hotpot_dev_fullwiki_v1.json
   ```
   **解决**: 确保 HotpotQA 数据文件在 `data/` 目录下

2. **API 错误**
   ```
   LLM generation failed: ...
   ```
   **解决**: 检查 API 密钥是否正确设置 (DEEPSEEK_API_KEY 等)

3. **导入错误**
   ```
   ModuleNotFoundError: No module named 'api_models'
   ```
   **解决**: 确保 `~/experiment/api_models.py` 存在,或修改 `src/llm_agent.py` 中的路径

## 扩展建议

### 添加新的评估指标

在 `src/evaluation.py` 中添加:

```python
def your_custom_metric(predictions, gold_answers):
    # 实现你的指标
    return score
```

### 自定义问题分解策略

修改 `src/decomposer_agent.py` 的 system_prompt:

```python
system_prompt = """You are a question decomposition specialist.
[Your custom instructions here]
"""
```

### 实现不同的信用分配方法

在 `src/reward_manager.py` 中添加:

```python
def allocate_custom_rewards(self, agents, contributions):
    # 实现自定义分配逻辑
    return rewards
```

## 参考

- HotpotQA 论文: https://arxiv.org/abs/1809.09600
- Shapley 值: https://en.wikipedia.org/wiki/Shapley_value
- 多智能体协作: 查看项目根目录的 `README.md`
