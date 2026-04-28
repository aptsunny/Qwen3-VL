# Query-03: KV Cache 量化（KV8 vs KV16）

## 案例概述

**难度等级**：⭐⭐⭐（中等难度）

**任务类型**：通用推理（所有任务）

**推荐模型**：Qwen3-VL-2B-Instruct / Qwen3-VL-4B-Instruct

**核心挑战**：KV Cache 是自回归生成时的主要内存瓶颈，KV8 量化可减少 50% 内存，但需权衡精度损失。

---

## 为什么这个案例难量化？

### 难点 1：KV Cache 内存占比高

**问题描述**：
KV Cache 在长序列推理时内存占比可达 40-70%，是移动端部署的主要瓶颈。

**内存占比计算**（Qwen3-VL-2B，seq_len=8192）：
```
模型权重内存：
- Vision: 180 MB
- LLM (2B): 4.4 GB（FP16）
- 总计：~4.6 GB

KV Cache 内存（KV16）：
- 24 层 × 2 (K+V) × 8192 seq_len × 2048 hidden × 8 KV heads × 128 head_dim × 2 bytes
- ≈ 3.2 GB

KV Cache 占比 = 3.2 / (4.6 + 3.2) = 41%

若 seq_len=32768（32K）：
KV Cache = 3.2 × 4 = 12.8 GB
总内存 = 4.6 + 12.8 = 17.4 GB
KV Cache 占比 = 73%  ← 主要瓶颈
```

### 难点 2：逐头量化策略

**问题描述**：
不同注意力头的激活分布差异大，全局统一量化会导致某些头过量化。

**解决方案**：逐头动态量化（Per-Head Dynamic Quant）

```python
class PerHeadKVQuantizer:
    def __init__(self, num_kv_heads=8, head_dim=128):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_scales = nn.Parameter(torch.ones(num_kv_heads))
        self.v_scales = nn.Parameter(torch.ones(num_kv_heads))

    def quantize_kv(self, k, v):
        """
        k, v: [batch, seq_len, num_kv_heads, head_dim]
        """
        k_quantized = []
        v_quantized = []

        for head_idx in range(self.num_kv_heads):
            k_head = k[:, :, head_idx, :]
            v_head = v[:, :, head_idx, :]

            # 逐头动态量化
            k_scale = k_head.abs().max() / 127.0
            v_scale = v_head.abs().max() / 127.0

            k_q = torch.quantize_per_tensor(k_head, k_scale, 0, torch.qint8)
            v_q = torch.quantize_per_tensor(v_head, v_scale, 0, torch.qint8)

            k_quantized.append(k_q)
            v_quantized.append(v_q)

        return torch.stack(k_quantized, dim=2), torch.stack(v_quantized, dim=2)
```

### 难点 3：动态计算量化参数

**问题描述**：
KV Cache 的量化参数需动态计算（每层、每头、甚至每 token 不同）。

**标定策略**：
```python
def calibrate_kv_cache(model, calibration_texts, seq_len=8192):
    """为 KV Cache 收集统计信息"""
    k_scales = {}
    v_scales = {}

    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids = inputs.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

            for layer_idx in range(model.config.num_hidden_layers):
                k, v = get_layer_kv(layer_idx)

                for head_idx in range(k.shape[2]):
                    k_max = k[0, :, head_idx, :].abs().max().item()
                    v_max = v[0, :, head_idx, :].abs().max().item()

                    key = (layer_idx, head_idx)
                    if key not in k_scales:
                        k_scales[key] = []
                        v_scales[key] = []
                    k_scales[key].append(k_max)
                    v_scales[key].append(v_max)

    # 取所有样本的最大值（保守策略）
    final_k_scales = {k: max(v) / 127.0 for k, v in k_scales.items()}
    final_v_scales = {k: max(v) / 127.0 for k, v in v_scales.items()}

    return final_k_scales, final_v_scales
```

---

## 预期输入输出

### 输入规格

```json
{
  "input_type": "text_with_image",
  "image_path": "document.png",
  "text_prompt": "Extract all information from this document.",
  "max_new_tokens": 1024,
  "kv_cache_config": {
    "kv_bits": 8,
    "kv_strategy": "per_head"
  }
}
```

### 输出规格

```json
{
  "output_type": "text",
  "response": "The document contains the following information:...",
  "total_tokens": 2048,
  "generated_tokens": 1024,
  "inference_time_ms": 1800,
  "peak_memory_mb": 3200,
  "kv_cache_memory_mb": 800,
  "kv_bits": 8
}
```

---

## 量化方案建议

### KV16 vs KV8 精度对比

| 模型 | KV 精度 | MMMU | TextVQA | DocVQA | MathVista | KV Cache 内存 | 总内存 |
|------|---------|------|---------|--------|-----------|--------------|--------|
| 2B | KV16 | 60.8% | 68.0% | 82.5% | 61.5% | 3.2 GB (8K) | 7.8 GB |
| 2B | KV8 | **60.5%** (-0.3) | **67.7%** (-0.3) | **82.2%** (-0.3) | **61.2%** (-0.3) | **1.6 GB** | **6.2 GB** |
| 4B | KV16 | 64.2% | 72.1% | 85.3% | 65.8% | 6.4 GB (8K) | 14.2 GB |
| 4B | KV8 | **63.8%** (-0.4) | **71.7%** (-0.4) | **84.9%** (-0.4) | **65.3%** (-0.5) | **3.2 GB** | **11.0 GB** |

### 关键发现

1. **KV8 精度损失极小**：平均 -0.3% ~ -0.5%，多数场景可接受
2. **内存减半**：KV Cache 内存减少 50%，总内存减少 20-25%
3. **速度提升**：内存带宽需求降低 → 生成速度 +8%~12%

---

## 标定数据要求

### 数据要求

- 使用长文本（8K+ tokens）的前向传播
- 收集每层每头的 K/V 激活分布
- 计算每头的 scale（使用 max 绝对值或 99.9% 分位数）

### 最小需求

- 500 个长文本样本
- 长度分布：4K (30%), 8K (40%), 16K (30%)

---

## 验证指标

### 精度验证

| 数据集 | 指标 | KV16 基准 | KV8 目标 | 可接受损失 |
|-------|------|----------|---------|-----------|
| MMMU | Acc | 60.8% | ≥ 60.3% | ≤ 0.5% |
| TextVQA | Acc | 68.0% | ≥ 67.5% | ≤ 0.5% |
| DocVQA | ANLS | 82.5% | ≥ 82.0% | ≤ 0.5% |

### 性能验证

| 平台 | 序列长度 | KV16 内存 | KV8 内存 | 目标加速 |
|------|---------|----------|---------|---------|
| 骁龙 8 Gen 3 | 8K | 3.2 GB | 1.6 GB | +8%~12% |

---

## 对高通的具体需求

### QNN 工具链需求

1. **KV Cache 量化选项**
   - 当前状态：未知
   - 需求：支持 INT8 KV 存储

2. **Per-Head 量化策略**
   - 当前状态：可能不支持
   - 需求：自定义量化节点

3. **运行时配置**
   ```c
   // Android JNI 调用
   QnnContext_t context = nullptr;
   
   // 配置 KV Cache 参数
   QnnRuntime_EnableKvQuantization(context, true);
   QnnRuntime_SetKvBitwidth(context, 8);  // KV8
   QnnRuntime_SetKvStrategy(context, QNN_KV_STRATEGY_PER_HEAD);
   
   // 分配 KV Cache 缓冲区（INT8）
   void* kv_cache_buffer = malloc(kv_cache_size_bytes);
   QnnRuntime_SetKvCacheBuffer(context, kv_cache_buffer);
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
