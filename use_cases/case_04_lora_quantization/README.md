# Case-04: LoRA + 量化融合

## 案例概述

**难度等级**：⭐⭐⭐⭐（高难度）

**任务类型**：领域定制化推理

**推荐模型**：Qwen3-VL-2B-Instruct / Qwen3-VL-4B-Instruct

**核心挑战**：LoRA 微调后的模型与量化存在兼容性问题，需要解决适配器融合和权重合并的技术难题。

---

## 为什么这个案例难量化？

### 难点 1：量化后 LoRA 失效

**问题描述**：
PTQ 将权重从 FP16 转为 INT4/INT8，但 LoRA 适配器是 FP16，两者不兼容。

**现象**：
```python
# 量化后推理
output = quant(W) × x

# 加上 LoRA
output = (quant(W) + ΔW) × x
# 问题：ΔW 是 FP16，需要先 dequantize 再相加
# 性能损失：每次推理都要 dequantize 权重 → 速度下降 30-50%
```

### 难点 2：权重融合策略

**问题描述**：
LoRA 权重合并到量化权重时，需要重新量化，可能引入额外误差。

**解决方案**：LoRA 权重融合

```python
def merge_lora_into_quantized_model(base_model, lora_model, quantizer):
    """
    将 LoRA 适配器合并到量化基础模型
    """
    for name, module in base_model.named_modules():
        if hasattr(module, 'weight') and name in lora_model.peft_config:
            # 获取 LoRA 参数
            lora_a = lora_model.peft_config[name].lora_A.weight  # [r, in_dim]
            lora_b = lora_model.peft_config[name].lora_B.weight  # [out_dim, r]

            # 计算 ΔW = B × A
            delta_w = lora_b @ lora_a  # [out_dim, in_dim]

            # 合并到基础权重（需先 dequantize）
            w_fp16 = module.weight.dequantize().float()
            w_merged = w_fp16 + delta_w * (lora_alpha / lora_r)

            # 重新量化
            w_quantized = quantizer.quantize(w_merged.half())

            # 替换权重
            module.weight = w_quantized

    return base_model
```

### 难点 3：多 LoRA 切换

**问题描述**：
如果需要动态切换多个 LoRA 适配器，每次都需要重新量化，部署复杂。

**解决方案**：动态解耦

```c
// Android 端推理
void inference_with_lora(uint8_t* input, char* prompt, int lora_id) {
    // 1. 加载基础量化模型
    QnnModel_t base_model = load_quantized_model("qwen3vl_base_q4.kv8.so");

    // 2. 加载 LoRA 适配器（FP16，小文件 ~10 MB）
    LoRAAdapter* lora = load_lora_adapter(lora_id);

    // 3. 运行时融合（每层）
    for (int layer = 0; layer < 24; layer++) {
        // 基础层推理（INT8）
        Tensor_t base_output = qnn_layer_inference(layer, input);

        // LoRA 分支（FP16，仅对特定层）
        if (lora->target_layers[layer]) {
            Tensor_t lora_output = lora_forward(lora, layer, input);
            // 融合：需要临时 dequantize
            Tensor_t base_fp16 = dequantize(base_output);
            Tensor_t merged = base_fp16 + lora_output;
            input = requantize(merged);  // 转回 INT8 传给下一层
        }
    }
}
```

**性能影响**：
- 额外 dequantize/requantize 开销：+15-20% 延迟
- LoRA 文件小（10-50 MB），可动态下载

---

## 预期输入输出

### 输入规格

```json
{
  "input_type": "image_with_text",
  "image_path": "medical_scan.png",
  "text_prompt": "Analyze this medical image and identify any abnormalities.",
  "lora_config": {
    "lora_id": "medical_v1",
    "lora_path": "lora_adapters/medical_v1.safetensors",
    "merge_strategy": "runtime"
  },
  "max_new_tokens": 512
}
```

### 输出规格

```json
{
  "output_type": "text",
  "response": "The medical image shows signs of inflammation in the upper right quadrant. Further examination is recommended...",
  "lora_applied": true,
  "lora_id": "medical_v1",
  "total_tokens": 1024,
  "inference_time_ms": 1200,
  "peak_memory_mb": 4500
}
```

---

## 量化方案建议

### 方案对比

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **静态合并** | 推理性能最优 | 无法动态切换 | 固定任务 |
| **动态解耦** | 可动态切换 | 性能损失 15-20% | 多任务平台 |
| **QALoRA** | 精度最高 | 训练成本高 | 高精度需求 |

### 推荐：静态合并（量产）

**流程**：
```
训练 LoRA → 合并权重 → 重新量化 → 部署
```

**步骤**：
```bash
# 1. 基础模型量化（PTQ 获取初始量化参数）
python quantize.py --model Qwen3-VL-2B --method ptq

# 2. QALoRA 微调（同时优化 LoRA + 量化参数）
python train_qa_lora.py \
  --base_model ./quantized_qwen3vl \
  --lora_r 8 \
  --lora_alpha 16 \
  --quant_lr 1e-4 \
  --lora_lr 1e-4 \
  --epochs 3 \
  --dataset coco_caption

# 3. 导出（合并 LoRA + 固化量化）
python export_merged.py \
  --qa_lora_checkpoint ./checkpoint \
  --output qwen3vl_2b_qa_lora_quantized.onnx
```

---

## 标定数据要求

### 数据要求

- 必须包含 LoRA 微调时的数据分布
- 建议：LoRA 训练集的 10-20% 用于标定

### 标定策略

```python
# 1. 先训练 LoRA（FP16）
lora_model = train_lora(
    base_model="Qwen3-VL-2B-Instruct",
    dataset="custom_dataset",
    lora_r=8,
    epochs=3
)

# 2. 使用 LoRA 模型的输出分布进行标定
calibration_data = sample_from_training_dataset(lora_model, num_samples=1000)

# 3. 计算量化参数（考虑 LoRA 的影响）
quantizer = Quantizer()
quantizer.calibrate(lora_model, calibration_data)

# 4. 合并并量化
merged_model = merge_lora(base_model, lora_model)
quantized_model = quantizer.quantize(merged_model)
```

---

## 验证指标

### 精度验证

| 任务类型 | FP16 LoRA | 量化后 | 可接受损失 |
|---------|----------|--------|-----------|
| **领域任务** | 85.0% | ≥ 83.0% | ≤ 2.0% |
| **基础任务** | 60.8% | ≥ 59.5% | ≤ 1.3% |

### 性能验证

| 方案 | 延迟 | 内存 | 适用性 |
|------|------|------|--------|
| **静态合并** | +0% | +5% | ✅ 量产 |
| **动态解耦** | +15-20% | +10% | ⚠️ 平台 |

---

## 对高通的具体需求

### QNN 工具链需求

1. **LoRA 动态加载**
   - 当前状态：不支持
   - 需求：运行时权重融合

2. **LoRA 融合工具**
   - 当前状态：无
   - 需求：自动合并并重新量化

3. **多 LoRA 内存管理**
   - 当前状态：无
   - 需求：多适配器切换策略

---

## 相关文件

- [推理脚本](./inference_script.py)
- [预期输入输出](./expected_input_output.json)
- [标定数据规格](./calibration_data_spec.md)

---

## 参考文档

- [Qwen3-VL 官方任务支持总结](../Qwen3VL_官方任务支持总结.md)
- [Qwen3-VL 困难量化案例指南](../Qwen3VL_困难量化案例_高通芯片部署指南.md)
- [Qwen-VL Finetune](../qwen-vl-finetune/)
