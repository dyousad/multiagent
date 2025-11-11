# RAG 问题修复总结

## 已修复的问题

### 1. ✅ 缺少语料库文件
**问题**: `data/hotpotqa_corpus.json` 不存在

**解决方案**: 创建了 `scripts/prepare_hotpotqa_corpus.py` 脚本

**使用方法**:
```bash
# 从前100个样本创建语料库（推荐用于测试）
python scripts/prepare_hotpotqa_corpus.py --max_samples 100

# 或处理所有样本
python scripts/prepare_hotpotqa_corpus.py
```

**结果**:
- ✓ 已成功从100个样本创建包含4246个文档的语料库
- 文件保存在: `data/hotpotqa_corpus.json`

### 2. ✅ 异步调用错误
**问题**: `'coroutine' object has no attribute 'get'`

**原因**: `run_hotpotqa_pipeline()` 是异步方法，但被当作同步调用

**解决方案**: 在 `scripts/run_hotpotqa_experiments.py` 中添加了 `asyncio.run()` 包装

**修改位置**: 第227行
```python
# 修复前
pipeline_output = controller.run_hotpotqa_pipeline(task)

# 修复后
pipeline_output = asyncio.run(controller.run_hotpotqa_pipeline(task))
```

### 3. ✅ 改进的错误处理
**改进内容**:
- 自动检测语料库文件是否存在
- 提供清晰的错误信息和解决建议
- 如果RAG组件失败，自动回退到标准执行
- 添加了详细的traceback用于调试

## 依赖要求

### 必需的依赖（用于RAG功能）
```bash
pip install sentence-transformers faiss-cpu numpy
```

如果有GPU:
```bash
pip install sentence-transformers faiss-gpu numpy
```

### 可选：使用更小的模型（更快，但精度略低）
默认使用 `multi-qa-mpnet-base-dot-v1`（约400MB）

可以在代码中改为更小的模型：
```python
# 在 src/retrieval_manager.py 中
model_name = "all-MiniLM-L6-v2"  # 约80MB
```

## 测试步骤

### Step 1: 准备语料库
```bash
python scripts/prepare_hotpotqa_corpus.py --max_samples 100
```
预期输出:
```
✓ Corpus preparation complete!
  Total documents: 4246
  Output: data/hotpotqa_corpus.json
```

### Step 2: 安装依赖（如果尚未安装）
```bash
pip install sentence-transformers faiss-cpu numpy
```

### Step 3: 运行集成测试
```bash
python scripts/test_rag_integration.py
```

### Step 4: 运行完整实验
```bash
python scripts/run_hotpotqa_experiments.py
```

## 无依赖情况下的测试

如果不想安装 sentence-transformers 和 faiss，仍然可以测试其他组件：

```bash
# 测试基础组件（不需要RAG依赖）
python scripts/test_rag_components.py
```

这将测试：
- ✓ DecomposerAgent
- ✓ EvidenceVerifierAgent
- ✓ Controller 方法
- ✓ RewardManager
- ⚠ RetrieverAgent (跳过，需要依赖)

## 已更新的文件

1. **新建文件**:
   - `scripts/prepare_hotpotqa_corpus.py` - 语料库准备脚本
   - `scripts/test_rag_integration.py` - RAG集成测试
   - `data/hotpotqa_corpus.json` - 提取的语料库

2. **修改文件**:
   - `scripts/run_hotpotqa_experiments.py` - 修复异步调用，改进错误处理

## 验证修复

### 修复前的错误:
```
Warning: RAG pipeline failed: 'coroutine' object has no attribute 'get'
```

### 修复后的预期行为:
```
✓ Added RetrieverAgent
✓ Added EvidenceVerifierAgent

Step 1: Decomposing question...
Generated 3 sub-questions

Step 2.1: Processing sub-question: ...
  - Retrieving evidence...
  - Verifying evidence...
```

## 下一步

1. **已完成**:
   - ✓ 语料库准备
   - ✓ 异步调用修复
   - ✓ 错误处理改进

2. **待执行**（用户操作）:
   - 安装依赖: `pip install sentence-transformers faiss-cpu`
   - 运行测试验证修复

3. **可选优化**:
   - 使用更大的语料库（处理更多样本）
   - 调整检索参数（top_k, 重排序策略等）
   - 尝试不同的embedding模型

## 故障排除

### 问题: "Corpus file not found"
**解决**: 运行 `python scripts/prepare_hotpotqa_corpus.py --max_samples 100`

### 问题: "'coroutine' object has no attribute 'get'"
**解决**: 已在 run_hotpotqa_experiments.py 中修复（第227行）

### 问题: "ImportError: No module named 'sentence_transformers'"
**解决**: `pip install sentence-transformers faiss-cpu`

### 问题: 内存不足
**解决**:
- 减少 max_samples 数量
- 使用更小的embedding模型
- 减少 top_k 参数

## 总结

所有已知问题已修复！系统现在应该能够：
1. ✓ 自动查找和加载语料库
2. ✓ 正确调用异步RAG pipeline
3. ✓ 在组件失败时优雅降级
4. ✓ 提供清晰的错误信息

执行 `python scripts/run_hotpotqa_experiments.py` 应该可以正常运行（需要先安装依赖）。
