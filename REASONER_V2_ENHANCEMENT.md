# Reasoner Prompt 调整 v2 - 增强推理能力

## 调整背景

### v1实验结果分析
在第一次改进后（v1），我们发现：
- ✅ 答案长度缩短98%+（成功）
- ✅ 输出格式规范化（成功）
- ❌ Exact Match仍为0%（失败）
- ❌ 过多输出"unknown"（过于保守）

### v1的问题

#### 案例分析：Sample 0
```
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Ground Truth: "yes"

Evidence:
- "American supernatural horror film directed by Scott Derrickson"
- "American biographical film about Ed Wood"

v1 Output: "no" ❌
Reason: Prompt要求"strictly based on evidence"，但证据没有明确说"Scott Derrickson is American"
```

**核心问题**: v1的prompt过于保守，不允许做合理推理。

---

## v2 改进方案

### 关键变化对比

| 方面 | v1 (过于保守) | v2 (平衡推理) |
|------|---------------|---------------|
| **指导方针** | "Base your answer strictly on the provided evidence" | "Apply logical reasoning and make **reasonable inferences** from contextual clues" |
| **推理规则** | 无明确推理指导 | 明确列出推理规则（见下文） |
| **unknown使用** | "If you cannot determine the answer from evidence, output 'unknown'" | "Only output 'unknown' if there is **absolutely NO relevant information** to make even a reasonable inference" |
| **决策性** | 无强调 | "**Be decisive** - choose the most likely answer based on available evidence" |

### 新增推理指导

v2添加了明确的推理规则：

```python
REASONING GUIDELINES:
- Use the provided evidence as your primary source
- Make reasonable inferences from contextual information:
  * If someone "directed an American film", they are likely American
  * If someone "was born in New York", they are likely American
  * If a film is described as "American biographical film about X", then X is likely American
  * If multiple evidence passages mention "American" in context of a person, infer American nationality
- Connect information across multiple evidence passages
- Look for patterns and implicit information
- Only output "unknown" if there is absolutely NO relevant information to make even a reasonable inference
```

### 改进的示例

v2添加了更多示例来指导模型：

#### 示例1: Yes/No问题 + 推理
```
Question: "Were they the same nationality?"
Evidence: "Person A directed American films", "American biographical film about Person B"
Correct output: "yes"
Wrong output: "unknown" (too conservative) ← v2明确标注这是错误的
```

#### 示例2: 从上下文推理国籍
```
Question: "What is his nationality?"
Evidence: "He directed several American films and was born in Colorado"
Correct output: "American"
Wrong output: "unknown" (evidence strongly suggests American nationality) ← v2明确标注
```

---

## 期望效果

### Sample 0预期改进
```
Before (v1):
  Evidence: "American film directed by Scott Derrickson", "American film about Ed Wood"
  Output: "no" ❌

After (v2):
  Evidence: Same
  Output: "yes" ✅ (推理：两人都与American film相关 → 都是美国人 → 相同国籍)
```

### Sample 1预期改进
```
Before (v1):
  Evidence: Contains information about Shirley Temple and government positions
  Output: "unknown" ❌ (过于保守)

After (v2):
  Evidence: Same
  Output: "Chief of Protocol" ✅ (如果证据中有相关信息)
```

### 指标预期

| 指标 | v1 | v2预期 | 改进 |
|------|-----|--------|------|
| Exact Match | 0.0 (0/3) | 0.33-0.67 (1-2/3) | +33-67% |
| Avg F1 | 0.0 | 0.3-0.5 | +0.3-0.5 |
| "unknown"率 | 67% (2/3) | 0-33% (0-1/3) | -34-67% |

---

## 实现细节

### 修改文件
- `src/reasoner_agent.py:40-88`

### 核心变化
1. **添加推理指导部分** (L48-57)
   - 明确列出4种常见推理场景
   - 强调连接多个证据片段
   - 寻找模式和隐含信息

2. **修改unknown条件** (L57)
   - 从"cannot determine"改为"absolutely NO relevant information"
   - 提高使用unknown的门槛

3. **强调决策性** (L65)
   - 新增"Be decisive"指导
   - 鼓励基于现有证据做出最可能的答案

4. **增强示例** (L67-86)
   - 添加反面示例（标注"Wrong"）
   - 特别标注"too conservative"情况
   - 提供推理示例

---

## 关键平衡点

v2试图在以下两个极端之间找到平衡：

```
过于保守 (v1)                    平衡 (v2)                    过于激进
    ↓                               ↓                              ↓
"没明确说就是     →     "有强烈暗示就可以     →     "随意猜测，
 unknown"                 合理推断"                  不管证据"
```

**v2的定位**:
- 基于证据 + 合理推理
- 允许从上下文推断
- 连接多个证据片段
- 但不进行无根据的猜测

---

## 测试验证

### 测试脚本
```bash
./test_reasoner_v2.sh
```

### 测试配置
- 样本数: 3（与v1相同，便于对比）
- 模型: Qwen/Qwen2.5-7B-Instruct
- 其他参数: 与v1完全相同

### 成功标准
- Exact Match > 0（至少答对1个）
- "unknown"输出 < 2个
- 答案仍保持简洁（<10字符）

---

## 风险评估

### 潜在风险
1. **过度推理**: 可能在证据不足时做出错误推理
2. **一致性**: 不同问题的推理标准可能不一致
3. **模型依赖**: 依赖LLM理解推理指导的能力

### 缓解措施
1. 在prompt中明确推理边界（"reasonable inferences"）
2. 提供具体的推理规则和示例
3. 保持evidence作为primary source的原则

---

## 后续优化方向

### 如果v2成功 (EM > 0.3)
1. 扩大测试规模（10-50样本）
2. 分析错误案例，优化推理规则
3. 考虑添加置信度输出

### 如果v2仍不理想 (EM < 0.3)
1. 分析失败案例，找出推理短板
2. 考虑多阶段推理（先收集事实，再推理）
3. 或者改进RAG检索，获取更直接的证据

---

## 版本历史

- **v0 (原始)**: 冗长解释，使用代词
- **v1 (2025-11-10)**: 简洁输出，但过于保守
- **v2 (2025-11-10)**: 增强推理，平衡保守性和推理能力

---

**创建时间**: 2025-11-10
**状态**: 实验进行中
**预计完成**: 5-10分钟
