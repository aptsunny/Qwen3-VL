# Case-02: 8K 长上下文推理

## 案例概述

**难度等级**：⭐⭐⭐⭐（高难度）

**任务类型**：Long Document Understanding（长文档理解）

**推荐模型**：Qwen3-VL-2B-Instruct / Qwen3-VL-4B-Instruct

**核心挑战**：长文档/长视频需要处理 8K-256K tokens，KV Cache 成为主要内存瓶颈，注意力权重数值稳定性成为关键问题。

---

## 为什么这个案例难量化？

### 难点 1：KV Cache 内存爆炸

**问题描述**：
长上下文推理时，KV Cache 存储所有历史 token 的 Key/Value，内存占用极大。

**内存计算**：
```
KV Cache 内存（每层）= 2 × seq_len × hidden_size × num_heads × head_dim × dtype_bytes

Qwen3-VL-2B:
- layers = 24
- hidden_size = 2048
- num_attention_heads = 16
- head_dim = 128
- num_kv_heads = 8（GQA）

KV16（FP16）内存：
单层 KV = 2 × seq_len × 8 × 128 × 2 bytes = 4096 × seq_len bytes
24 层总内存 = 98,304 × seq_len bytes

示例：
seq_len = 8,192（8K）→ 98,304 × 8192 ≈ 786 MB（KV16）
seq_len = 32,768（32K）→ 786 × 4 = 3.1 GB
seq_len = 262,144（256K）→ 786 × 32 = 25.1 GB（无法部署）
```

**移动端可行配置**（骁龙 8 Gen 3，8GB RAM）：
```
最大 seq_len = 16,384（16K）→ KV16 需 1.5 GB
剩余内存给模型权重、激活、系统 → 可行

若使用 KV8（INT8）：
内存减半 → seq_len=32,768（32K）可行
```

### 难点 2：注意力权重数值稳定性

**问题描述**：
长序列的 softmax 注意力权重范围极大（极小概率事件被忽略），INT8 量化后可能丢失。

**现象**：
- 短序列（<4K）：精度损失 < 0.5%
- 长序列（8K-32K）：精度损失 1-2%
- 超长序列（>64K）：精度损失 > 3%（可能崩溃）

**根本原因**：
```python
# 长序列注意力权重分布
attn_weights = softmax(Q @ K.T / sqrt(d))

# 问题：长序列中，大部分注意力权重接近 0，少数接近 1
# INT8 量化后，接近 0 的权重被截断 → 信息丢失

# 示例
seq_len = 8192
attn_weights.min() = 1e-6  # 极小值
attn_weights.max() = 0.95  # 极大值
# INT8 量化范围：[-128, 127] / 128 → 最小分辨率 ~0.008
# 1e-6 被截断为 0 → 丢失信息
```

### 难点 3：位置编码量化误差

**问题描述**：
Qwen3-VL 使用 Interleaved-MRoPE（多模态相对位置编码），位置 ID 的旋转计算对精度敏感。

**MRoPE 公式**：
```
q_rotated = q * cos(mrope) + rotate_half(q) * sin(mrope)
其中 mrope 由 time/width/height 三个维度的位置 ID 计算
```

**量化影响**：
- cos/sin 值域 [-1, 1]，INT8 量化后精度损失显著
- 位置误差累积 → 长序列注意力偏移

### 难点 4：梯度检查点与量化冲突

**问题描述**：
长上下文训练时使用梯度检查点节省内存，但量化后重新计算激活会导致数值不一致。

**冲突场景**：
```python
# 训练脚本
if training_args.gradient_checkpointing:
    model.enable_input_require_grads()  # 启用梯度检查点

# 量化后：前向是量化算子，但检查点保存的是 FP16 激活
# 反向时重新计算 → 数值不匹配 → 训练不稳定
```

---

## 预期输入输出

### 输入规格

```json
{
  "input_type": "long_document",
  "document_type": "multi_page_pdf",
  "pages": [
    {"page_id": 1, "image_path": "page1.png"},
    {"page_id": 2, "image_path": "page2.png"},
    {"page_id": 3, "image_path": "page3.png"}
  ],
  "text_prompt": "Summarize the key points from all pages and answer: What is the main conclusion?",
  "context_length": 8192,
  "max_new_tokens": 512
}
```

### 输出规格

```json
{
  "output_type": "text",
  "response": "Based on the analysis of all pages, the document discusses climate change impacts. The main conclusion is that immediate action is required to reduce carbon emissions...",
  "total_tokens": 8704,
  "context_tokens": 8192,
  "generated_tokens": 512,
  "inference_time_ms": 2500,
  "peak_memory_mb": 4500,
  "kv_cache_memory_mb": 1570
}
```

---

## 量化方案建议

### 推荐配置

| 模块 | 量化方案 | 精度损失 | 内存减少 | 推荐 |
|------|---------|---------|---------|------|
| **Vision Encoder** | W8A16 | -0.5% | 1.5× | ✅ **推荐** |
| **LLM Weights** | W4A16 | -1.0% | 2.2× | ✅ **推荐** |
| **KV Cache** | KV16 | 0% | 1× | ⚠️ 内存大 |
| **KV Cache** | KV8 | -0.3% | 2× | ✅ **必须** |

### 最佳组合

```
Vision W8A16 + LLM W4A16 + KV8
- 总精度损失：~1.3%
- KV Cache 内存减半
- 支持 16K-32K 上下文
```

### 分段量化策略

```python
class SegmentAwareQuantizer:
    def __init__(self, segment_length=2048):
        self.segment_length = segment_length  # 每段 2K tokens
        self.segment_scales = {}

    def quantize_sequence(self, hidden_states):
        seq_len = hidden_states.shape[0]
        num_segments = (seq_len + self.segment_length - 1) // self.segment_length

        quantized_segments = []
        for seg_idx in range(num_segments):
            start = seg_idx * self.segment_length
            end = min((seg_idx + 1) * self.segment_length, seq_len)
            segment = hidden_states[start:end]

            # 每段独立计算量化参数
            scale = segment.abs().max() / 127.0
            quantized_seg = torch.quantize_per_tensor(segment, scale, 0, torch.qint8)

            self.segment_scales[seg_idx] = scale
            quantized_segments.append(quantized_seg)

        return quantized_segments
```

---

## 标定数据要求

### 数据分布覆盖

| 任务类型 | 序列长度分布 | 数据来源 | 样本数 |
|---------|------------|---------|--------|
| **长文档 QA** | 4K (30%), 8K (40%), 16K (30%) | PubMedQA, arXiv, Legal documents | 500 样本 |
| **多图推理** | 8-16 张图（每图 256 tokens） | COCO, VisualGenome | 300 样本 |
| **视频理解** | 16-64 帧（每帧 256 tokens） | MSRVTT, ActivityNet | 200 样本 |
| **代码理解** | 8K-32K tokens（长代码文件） | GitHub repositories | 300 样本 |

### 标定流程

```python
# 使用长度分层的采样策略
def sample_calibration_batch(dataset, target_lengths):
    """
    target_lengths: 按比例采样，如 {4096: 0.3, 8192: 0.4, 16384: 0.3}
    """
    batch = []
    for length, ratio in target_lengths.items():
        n_samples = int(total_samples * ratio)
        samples = filter_by_seq_len(dataset, length, tolerance=0.2*length)
        batch.extend(random.sample(samples, n_samples))
    return batch
```

---

## 验证指标

### 精度验证

| 数据集 | 指标 | FP16 基准 | 量化目标 | 可接受损失 |
|-------|------|----------|---------|-----------|
| **LongBench** | 平均分 | 45.2% | ≥ 44.0% | ≤ 1.2% |
| **DocVQA** | ANLS | 82.5% | ≥ 81.0% | ≤ 1.5% |
| **NarrativeQA** | F1 | 35.8% | ≥ 34.5% | ≤ 1.3% |

### 性能验证

| 平台 | 序列长度 | 目标延迟 | 目标内存 |
|------|---------|---------|---------|
| 骁龙 8 Gen 3 | 8K | ≤ 2.5 s | ≤ 4 GB |
| 骁龙 8 Gen 3 | 16K | ≤ 5.0 s | ≤ 5 GB |
| Cloud AI 100 | 32K | ≤ 1.0 s | - |

---

## 对高通的具体需求

### QNN 工具链需求

1. **动态 shape 编译**
   - 当前状态：部分支持
   - 需求：完全动态 batch/seq_len 支持

2. **Per-Head KV 量化**
   - 当前状态：可能不支持
   - 需求：逐头 KV Cache 量化策略

3. **位置编码保留 FP16**
   - 当前状态：未知
   - 需求：MRoPE 位置编码保留 FP16 计算

4. **分段量化支持**
   - 当前状态：不支持
   - 需求：支持序列分段独立量化

### 标定工具需求

```python
# 期望的 API 设计
qnn.calibrate_long_context(
    model=model,
    data=long_text_pairs,
    seq_lengths=[4096, 8192, 16384],
    kv_quantization="per_head",
    segment_length=2048
)
```

---

## 相关文件

- [推理脚本](./inference_script.py)
- [预期输入输出](./expected_input_output.json)
- [标定数据规格](./calibration_data_spec.md)

---

## 参考文档

- [Qwen3-VL 官方任务支持总结](../Qwen3VL_官方任务支持总结.md)
- [Qwen3-VL 困难量化案例指南](../Qwen3VL_困难量化案例_高通芯片部署指南.md)
- [Long Document Understanding Cookbook](../cookbooks/long_document_understanding.ipynb)
