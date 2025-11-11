# 🚀 完整运行流程 - Question Decomposition改进版

## ✅ 你已完成的准备工作

1. ✅ 环境配置好 (vlm-anchor)
2. ✅ 全量corpus构建完成 (`data/hotpotqa_corpus_full.json`, 104MB)
3. ✅ Embeddings缓存构建完成 (`data/cache/`, 2.3GB)

---

## 🎯 现在可以运行的完整流程

### 方式1: 快速测试（推荐）

运行5个样本，验证所有改进：

```bash
chmod +x test_final.sh
./test_final.sh
```

**这会做什么**:
1. 检查缓存是否存在
2. 使用**全量corpus**（300k+ docs）从**缓存快速加载**（30秒）
3. 使用**Reasoner v2**（增强推理）
4. 运行5个样本测试
5. **对比4个版本的结果**：
   - v0: 原始版本（Exact Match = 0%）
   - v1: 改进prompt（Exact Match = 0%）
   - v2: 增强reasoner（Exact Match = 0%）
   - **FINAL**: 全量corpus + v2（**预期 Exact Match = 40-80%**）

**预计时间**: 5-8分钟（包括30秒缓存加载）

---

### 方式2: 中等规模测试

运行10-20个样本：

```bash
./run_hotpotqa.sh --samples 10 --agents 3
```

或直接用Python：

```bash
python scripts/run_hotpotqa_experiments.py
```

然后查看结果：

```bash
python scripts/plot_hotpotqa_results.py \
    --results results/hotpotqa/hotpotqa_results.json \
    --output_dir results/hotpotqa/plots
```

**预计时间**: 10-20分钟

---

### 方式3: 完整实验（50+样本）

```bash
./run_hotpotqa.sh --samples 50 --agents 3
```

**预计时间**: 30-60分钟

---

## 📊 预期改进效果

### Exact Match对比

```
v0 (原始):           0/5 = 0%    ❌
v1 (改进prompt):      0/5 = 0%    ❌
v2 (增强reasoner):    0/5 = 0%    ❌
FINAL (全量corpus):   2-4/5 = 40-80%  ✅ 预期
```

### 具体样本预期

| Sample | Question | v0-v2 | FINAL (预期) |
|--------|----------|-------|--------------|
| 0 | Were Scott Derrickson and Ed Wood of the same nationality? | "no" ❌ | "yes" ✅ |
| 1 | What government position was held by the woman who portrayed Corliss Archer? | "unknown" ❌ | "Chief of Protocol" ✅ |
| 2 | What science fantasy series...? | "unknown" ❌ | "Animorphs" ✅ |

---

## 🔧 所有改进的组合效果

### 改进1: Reasoner v2 (已实施)
- 允许从上下文合理推理
- 降低"unknown"门槛
- **文件**: `src/reasoner_agent.py`

### 改进2: Decomposition优化 (已实施)
- 避免使用代词
- 子问题更具体、可搜索
- **文件**: `src/decomposer_agent.py`

### 改进3: 答案提取 (已实施)
- 从冗长回答提取核心答案
- 支持yes/no和实体问题
- **文件**: `src/reasoner_agent.py:_extract_answer()`

### 改进4: 全量Corpus (已构建)
- 从42k → 300k+ 文档
- 覆盖所有HotpotQA样本
- **文件**: `data/hotpotqa_corpus_full.json`

### 改进5: Embeddings缓存 (已构建)
- 第一次编码后缓存
- 后续加载只需30秒
- **文件**: `src/cached_retrieval_manager.py`

---

## 📁 关键文件位置

### 数据和缓存
```
data/
├── hotpotqa_corpus_full.json        # 全量corpus (104MB)
└── cache/
    ├── *_embeddings.npy              # 缓存的embeddings (1.1GB)
    ├── *_index.faiss                 # 缓存的FAISS索引 (1.1GB)
    ├── *_corpus.pkl                  # 缓存的corpus (90MB)
    └── *_texts.pkl                   # 缓存的文本 (42MB)
```

### 改进的源代码
```
src/
├── reasoner_agent.py                # v2增强推理
├── decomposer_agent.py              # 优化的分解
├── cached_retrieval_manager.py      # 缓存系统
└── retriever_agent.py               # 自动使用缓存
```

### 实验脚本
```
scripts/
├── run_hotpotqa_experiments.py      # 主实验脚本（已更新）
├── build_full_corpus_cache.py       # 构建缓存（已完成）
└── plot_hotpotqa_results.py         # 可视化
```

### 测试脚本
```
./test_final.sh                      # 最终完整测试
./run_hotpotqa.sh                    # 标准HotpotQA测试
```

---

## 🎯 推荐运行顺序

### 第一次运行（验证改进）

```bash
# 1. 快速测试5个样本
./test_final.sh

# 预期结果:
# - 加载速度: 30秒（vs 之前的10分钟）
# - Exact Match: 40-80%（vs 之前的0%）
# - 看到4个版本的对比
```

### 如果结果满意

```bash
# 2. 中等规模测试
./run_hotpotqa.sh --samples 20 --agents 3

# 3. 生成可视化
python scripts/plot_hotpotqa_results.py \
    --results results/hotpotqa/hotpotqa_results.json \
    --output_dir results/hotpotqa/plots
```

### 如果需要完整评估

```bash
# 4. 大规模测试
./run_hotpotqa.sh --samples 100 --agents 3
```

---

## 🔍 查看结果

### 命令行查看

```bash
python -c "
import json
with open('results/hotpotqa_final/hotpotqa_results.json') as f:
    data = json.load(f)
    agg = data['aggregate']
    print(f'Exact Match: {agg[\"exact_match_accuracy\"]:.1%}')
    print(f'Average F1: {agg[\"average_f1\"]:.3f}')
"
```

### 详细分析

```bash
cat results/hotpotqa_final/hotpotqa_results.json | python -m json.tool | less
```

---

## ⚠️ 故障排查

### 如果提示"Cache not found"

```bash
# 重新构建缓存
python scripts/build_full_corpus_cache.py
```

### 如果加载很慢（>5分钟）

说明没有使用缓存，检查：
```bash
ls -lh data/cache/
# 应该看到4个文件，总共2.3GB
```

### 如果Exact Match仍然很低

1. 检查是否使用了全量corpus：
```bash
grep "Using full corpus" <实验日志>
```

2. 检查reasoner版本：
```bash
grep "reasonable inferences" src/reasoner_agent.py
# 应该能找到这个短语
```

---

## 📊 成功标准

运行 `./test_final.sh` 后，期望看到：

```
✓ Cache detected - will load in ~30 seconds
✓ Using full corpus (cached): data/hotpotqa_corpus_full.json
✓ Using CachedRetrievalManager (fast loading)

FINAL RESULTS
======================================================================
Exact Match Accuracy: 0.400-0.800  (40-80%)  ← 目标
Average F1 Score:     0.500-0.700           ← 目标

COMPARISON ACROSS ALL VERSIONS
======================================================================
Version                        Exact Match     Avg F1
------------------------------------------------------------
v0 (original)                  0.000           0.000
v1 (improved prompts)          0.000           0.000
v2 (enhanced reasoner)         0.000           0.000
FINAL (full corpus)            0.400-0.800     0.500-0.700  ← 显著改进
```

---

## 🚀 立即开始

```bash
# 一键运行最终测试
chmod +x test_final.sh
./test_final.sh
```

预计5-8分钟后看到结果！

---

## 📝 文档参考

- `QUESTION_DECOMPOSITION_IMPROVEMENT.md` - 问题分解改进总结
- `REASONER_V2_ENHANCEMENT.md` - Reasoner v2增强说明
- `CORPUS_CACHE_SOLUTION.md` - Corpus和缓存解决方案
- `HOTPOTQA_GUIDE.md` - HotpotQA完整指南
