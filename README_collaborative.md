# 纯协作模式多智能体系统配置指南

## 🎯 系统特点

✅ **解决了原有问题**:
- 消除了不公平的credit分配 (之前reasoner占主导26.58 vs 0.9)
- 所有agents平等参与 (不再有rag_前缀的虚拟输出)
- 支持角色权重 (重要角色获得适当加权)

✅ **新增功能**:
- 5个专业化角色协作
- 基于角色重要性的动态权重
- 公平性评估指标 (Shannon熵)

## ⚙️ 配置文件说明

### 1. config.yaml 配置

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
    decomposer: 1.2   # 问题分解很重要
    researcher: 1.0   # 基准权重
    analyst: 1.5      # 分析推理最重要
    reviewer: 0.8     # 质量检查中等重要
    synthesizer: 1.3  # 最终合成很重要
```

### 2. 运行方式

#### 方式A: 使用脚本
```python
from scripts.run_hotpotqa_experiments import run_hotpotqa_experiment

results = run_hotpotqa_experiment(
    num_agents=5,
    use_rag=False,  # 关键: 禁用RAG
    use_pure_collaborative=True,  # 启用纯协作
    use_dynamic_credit=True,
    role_weights={
        "decomposer": 1.2,
        "researcher": 1.0,
        "analyst": 1.5,
        "reviewer": 0.8,
        "synthesizer": 1.3
    }
)
```

#### 方式B: 命令行
```bash
python scripts/run_hotpotqa_experiments.py
```

## 📊 Credit分配原理

### 反事实分析算法
1. **基线分数**: 计算所有agents协作的分数
2. **反事实分数**: 逐个移除agent，计算剩余agents的分数
3. **边际贡献**: `credit = baseline_score - counterfactual_score`
4. **角色权重**: `final_credit = marginal_contribution × role_weight`

### 示例结果
```python
# 之前 (不公平)
Dynamic Credits: {
  'rag_What is Scott Derric': 0.96,
  'reasoner': 26.58,  # 不公平的主导
}

# 现在 (公平)
Dynamic Credits: {
  'agent_0 (decomposer)': 0.0247 × 1.2 = 0.0296,
  'agent_1 (researcher)': 0.0294 × 1.0 = 0.0294,
  'agent_2 (analyst)': 0.0227 × 1.5 = 0.0341,    # 最高 (权重大)
  'agent_3 (reviewer)': 0.0296 × 0.8 = 0.0237,
  'agent_4 (synthesizer)': 0.0195 × 1.3 = 0.0254
}
```

## 🔧 自定义配置

### 调整角色权重
```yaml
role_weights:
  decomposer: 1.0    # 降低问题分解权重
  researcher: 1.2    # 增加研究权重
  analyst: 2.0       # 大幅增加分析权重
  reviewer: 0.5      # 减少审核权重
  synthesizer: 1.5   # 增加合成权重
```

### 更改角色定义
```python
# 在 create_agents_with_rag() 中修改
role_prompts = {
    "decomposer": "你是问题分解专家...",
    "researcher": "你是信息研究员...",
    "analyst": "你是数据分析师...",
    "reviewer": "你是质量审核员...",
    "synthesizer": "你是答案综合者..."
}
```

## 📈 性能评估

### 公平性指标
- **Dynamic Entropy**: 越高=越公平 (目标: >1.5)
- **Credit Variance**: 越低=越均匀
- **Jain Fairness Index**: 接近1=最公平

### 示例分析代码
```python
def analyze_fairness(dynamic_credits):
    values = list(dynamic_credits.values())

    # Shannon熵 (公平性)
    entropy = -sum(p * log(p) for p in values if p > 0)

    # Jain公平指数
    sum_squares = sum(x**2 for x in values)
    fairness = (sum(values)**2) / (len(values) * sum_squares)

    return {
        "entropy": entropy,
        "fairness_index": fairness,
        "variance": statistics.variance(values)
    }
```

## 🚀 使用建议

### 1. 针对不同任务调整权重
- **创意任务**: 增加synthesizer权重
- **分析任务**: 增加analyst权重
- **研究任务**: 增加researcher权重

### 2. 监控公平性
```python
if dynamic_entropy < 1.0:
    print("⚠️ Credit分配不够公平，考虑调整权重")
elif dynamic_entropy > 2.0:
    print("✅ Credit分配很公平")
```

### 3. 调试技巧
```python
# 查看agent输出
for agent_id, output in agent_outputs.items():
    print(f"{agent_id}: {output[:100]}...")

# 分析credit分布
total_credit = sum(dynamic_credits.values())
for agent_id, credit in dynamic_credits.items():
    percentage = (credit / total_credit) * 100
    print(f"{agent_id}: {percentage:.1f}%")
```

## ⚡ 快速启动

```bash
# 1. 克隆并进入项目
cd multiagent

# 2. 测试纯协作模式
python test_collaborative.py

# 3. 运行完整实验
python test_enhanced.py

# 4. 查看结果
cat results/enhanced_collaborative/hotpotqa_results.json
```

这个系统现在提供了公平、可配置的多智能体协作，完全解决了之前RAG模式中credit分配不均的问题！