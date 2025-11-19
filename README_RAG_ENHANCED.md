# RAG-Enhanced Multi-Agent System - 使用指南

## 🎯 系统特性

### ✅ 已实现的核心功能

1. **角色化Agent命名**
   - ❌ 之前: `agent_0`, `agent_1`, `agent_2`
   - ✅ 现在: `decomposer`, `researcher`, `analyst`, `reviewer`, `synthesizer`

2. **公平的Credit分配**
   - ❌ 之前: 不公平的子问题级别分配 (`reasoner: 26.58` vs `rag_*: 0.9`)
   - ✅ 现在: Agent级别分配，支持角色权重

3. **RAG检索能力**
   - ❌ 之前: 只有专门的RAG agents
   - ✅ 现在: 每个agent都具备检索能力

4. **角色权重系统**
   - ✅ 可配置的角色重要性权重
   - ✅ 基于角色贡献的公平奖励分配

---

## 🚀 快速开始

### 方式1: 使用Shell脚本

```bash
# 赋予执行权限
chmod +x run_rag_enhanced.sh

# 运行实验
./run_rag_enhanced.sh
```

### 方式2: 直接运行Python脚本

```bash
python scripts/run_hotpotqa_experiments.py
```

### 方式3: 编程方式调用

```python
from scripts.run_hotpotqa_experiments import run_hotpotqa_experiment

results = run_hotpotqa_experiment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=10,
    num_agents=5,
    model_identifier="Qwen/Qwen2.5-7B-Instruct",
    use_rag_enhanced=True,  # 启用RAG增强模式
    role_weights={
        "decomposer": 1.2,
        "researcher": 1.0,
        "analyst": 1.5,      # 最高权重
        "reviewer": 0.8,
        "synthesizer": 1.3
    }
)
```

---

## ⚙️ 配置说明

### config.yaml 配置

```yaml
# Agent Configuration
agents:
  num_agents: 5
  roles:
    - decomposer    # 问题分解和分析
    - researcher    # 信息收集和检索
    - analyst       # 分析和推理
    - reviewer      # 质量控制和验证
    - synthesizer   # 答案合成和生成

  # 角色权重 (越高=越重要)
  role_weights:
    decomposer: 1.2   # 问题分解重要性
    researcher: 1.0   # 基准权重
    analyst: 1.5      # 分析最重要
    reviewer: 0.8     # 审核中等重要
    synthesizer: 1.3  # 合成很重要
```

---

## 📊 系统架构

### Agent角色职责

```
decomposer   (权重: 1.2)
  └─ 职责: 将复杂问题分解为子问题
  └─ RAG: 检索相关背景知识

researcher   (权重: 1.0)
  └─ 职责: 收集信息和证据
  └─ RAG: 检索具体事实数据

analyst      (权重: 1.5) ⭐ 最重要
  └─ 职责: 分析推理和得出结论
  └─ RAG: 检索支持性证据

reviewer     (权重: 0.8)
  └─ 职责: 质量控制和验证
  └─ RAG: 检索验证信息

synthesizer  (权重: 1.3)
  └─ 职责: 综合答案生成
  └─ RAG: 检索补充信息
```

### Credit分配算法

```python
# 1. 反事实分析
baseline_score = evaluate(final_answer, ground_truth)
for agent in agents:
    without_agent_score = evaluate(answer_without_agent, ground_truth)
    marginal_contribution = baseline_score - without_agent_score

# 2. 应用角色权重
weighted_credit = marginal_contribution × role_weight

# 3. 累积正向贡献
agent_credit += max(weighted_credit, 0)
```

---

## 📈 预期输出

### 成功运行示例

```
🚀 Running RAG-Enhanced Collaborative Mode
============================================================
Features:
  ✓ Role-based agent names
  ✓ Each agent has RAG retrieval
  ✓ Fair credit allocation with role weights
  ✓ Agent-level credit distribution
============================================================

--- Sample 1/10 ---
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Ground Truth: yes
Predicted: Yes, both were American.
Exact Match: True
F1 Score: 1.000

Dynamic Credits:
  decomposer: 0.18 (weight: 1.2, effectiveness: 0.15)
  researcher: 0.25 (weight: 1.0, effectiveness: 0.25)
  analyst: 0.38 (weight: 1.5, effectiveness: 0.25)
  reviewer: 0.12 (weight: 0.8, effectiveness: 0.15)
  synthesizer: 0.28 (weight: 1.3, effectiveness: 0.22)

📊 Experiment Summary
============================================================
Samples Processed: 10
Exact Match Accuracy: 0.800
Average F1 Score: 0.856

🎯 Credit Distribution (Role-Based)
--------------------------------------------------
decomposer   | Avg Credit: 0.2145 | Weight: 1.2 | Effectiveness: 0.1788
researcher   | Avg Credit: 0.2891 | Weight: 1.0 | Effectiveness: 0.2891
analyst      | Avg Credit: 0.4235 | Weight: 1.5 | Effectiveness: 0.2823
reviewer     | Avg Credit: 0.1456 | Weight: 0.8 | Effectiveness: 0.1820
synthesizer  | Avg Credit: 0.3124 | Weight: 1.3 | Effectiveness: 0.2403
```

---

## 🔧 自定义配置

### 调整角色权重

根据不同任务类型调整权重:

#### 创意任务
```yaml
role_weights:
  decomposer: 1.0
  researcher: 0.8
  analyst: 1.2
  reviewer: 0.6
  synthesizer: 1.8  # 增加创意合成权重
```

#### 分析任务
```yaml
role_weights:
  decomposer: 1.5  # 增加问题分解权重
  researcher: 1.2
  analyst: 2.0     # 大幅增加分析权重
  reviewer: 1.0
  synthesizer: 1.0
```

#### 研究任务
```yaml
role_weights:
  decomposer: 1.0
  researcher: 1.8  # 增加研究权重
  analyst: 1.3
  reviewer: 0.8
  synthesizer: 1.2
```

### 修改Agent数量

```yaml
agents:
  num_agents: 3  # 减少到3个agents
  roles:
    - researcher
    - analyst
    - synthesizer
```

---

## 🐛 故障排除

### 问题1: RAG检索失败

**症状**: `⚠ Retrieval manager initialization failed`

**解决方案**:
```bash
# 检查是否有corpus文件
ls data/hotpotqa_corpus_full.json

# 如果没有，构建缓存
python scripts/build_full_corpus_cache.py

# 或者系统会自动降级到无RAG模式
```

### 问题2: Credit全为0

**症状**: `Dynamic Credits: {'decomposer': 0.0, ...}`

**原因**: 答案与ground truth完全不匹配

**解决方案**: 检查模型输出质量，可能需要:
- 调整temperature参数
- 使用更强的模型
- 增加max_tokens

### 问题3: Agent命名错误

**症状**: 仍然显示`agent_0`而不是角色名

**解决方案**: 确保使用`use_rag_enhanced=True`参数

---

## 📁 结果文件

实验结果保存在: `results/hotpotqa_rag_enhanced/`

```
results/hotpotqa_rag_enhanced/
├── hotpotqa_results.json          # 完整结果
├── sample_0.json                   # 每个样本的详细结果
├── sample_1.json
└── ...
```

结果包含:
- 问题和答案
- 每个agent的输出
- Credit分配详情
- F1分数和Exact Match
- Dynamic Entropy (公平性指标)

---

## 🎯 核心改进总结

| 特性 | 之前 | 现在 |
|------|------|------|
| Agent命名 | `agent_0, agent_1...` | `decomposer, researcher, analyst...` |
| Credit分配单位 | 子问题级别 | Agent级别 |
| RAG能力 | 只有专门agents | 每个agent都有 |
| 角色权重 | 不支持 | 完全支持 |
| 公平性 | 低 (reasoner主导) | 高 (均衡分配) |
| 准确率 | 依赖LLM知识 | 可检索外部知识 |

---

## 📚 相关文件

- `src/rag_enhanced_agent.py` - RAG增强Agent实现
- `src/reward_manager.py` - 角色权重Credit分配
- `scripts/run_hotpotqa_experiments.py` - 主实验脚本
- `config.yaml` - 系统配置
- `test_rag_enhanced.py` - 测试脚本
- `README_collaborative.md` - 纯协作模式文档

---

## 💡 最佳实践

1. **首次运行**: 使用小样本量 (max_samples=5) 测试
2. **调整权重**: 根据任务特点调整角色权重
3. **监控公平性**: 检查Dynamic Entropy (>1.5为公平)
4. **分析结果**: 查看每个agent的effectiveness
5. **迭代优化**: 根据结果调整配置

---

## 🆘 获取帮助

如有问题,请检查:
1. 日志输出中的错误信息
2. `results/` 目录中的结果文件
3. 配置文件格式是否正确

---

**祝使用愉快! 🎉**