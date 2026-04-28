# Query-05: 多模态联合标定

## 案例概述

**难度等级**：⭐⭐⭐⭐（高难度）

**任务类型**：通用多模态推理

**推荐模型**：Qwen3-VL-2B-Instruct / Qwen3-VL-4B-Instruct

**核心挑战**：ViT 单独标定导致多模态任务量化损失大，需要 Vision+LLM 整体联合 calibration 方案。

---

## 为什么这个案例难量化？

### 难点 1：独立标定的缺陷

**问题描述**：
先前按 ViT 单独用 COCO 标定、再整体导出评估的方式在多模态任务上量化损失偏大。

**现象**：
```python
# 传统方式（错误）
vision_quant_params = calibrate_vision(vision_model, coco_images)
llm_quant_params = calibrate_llm(llm_model, text_data)

# 问题：Vision 和 LLM 的量化参数独立计算
# 多模态特征融合时，激活分布不匹配 → 精度下降 3-5%
```

### 难点 2：跨模态特征对齐

**问题描述**：
Vision 和 LLM 的特征分布差异大，独立量化会导致特征对齐失败。

**示例**：
```python
# Vision 特征分布
vision_features.mean() = 0.02
vision_features.std() = 0.15

# LLM 期望的输入分布
llm_expected.mean() = 0.05
llm_expected.std() = 0.20

# 独立量化后，分布偏移
# → 多模态融合层激活异常
# → 精度下降
```

### 难点 3：标定数据多样性

**问题描述**：
需要覆盖多种模态组合（图像+文本、视频+文本、多图+文本），标定数据准备复杂。

**数据要求**：
```python
calibration_suite = {
    "image_text": {
        "single_image": "COCO, 1000 samples",
        "multi_image": "VisualGenome, 500 samples",
    },
    "video_text": {
        "short_video": "MSRVTT, 200 samples (8 frames)",
        "long_video": "ActivityNet, 100 samples (32 frames)",
    },
    "document_text": {
        "single_page": "DocVQA, 300 samples",
        "multi_page": "PDF documents, 100 samples",
    }
}
```

---

## 预期输入输出

### 输入规格

```json
{
  "input_type": "multimodal",
  "modalities": ["image", "text"],
  "image_path": "sample_image.jpg",
  "text_prompt": "Describe this image in detail.",
  "calibration_config": {
    "joint_calibration": true,
    "modalities": ["vision", "text"],
    "num_samples": 1000
  }
}
```

### 输出规格

```json
{
  "output_type": "text",
  "response": "The image shows a busy city street with people walking...",
  "quantization_params": {
    "vision": {"bitwidth": 8, "granularity": "per_channel"},
    "llm": {"bitwidth": 4, "granularity": "per_channel"},
    "kv_cache": {"bitwidth": 8, "strategy": "per_head"}
  },
  "calibration_quality": {
    "vision_llm_alignment": 0.95,
    "activation_coverage": 0.92
  }
}
```

---

## 量化方案建议

### 联合标定流程

```python
class MultiModalJointCalibrator:
    def __init__(self, model):
        self.model = model
        self.vision_activations = {}
        self.llm_activations = {}
        self.fusion_activations = {}

        # 注册激活钩子
        for name, module in model.named_modules():
            if "vision" in name:
                module.register_forward_hook(self.get_vision_activation(name))
            elif "language_model" in name:
                module.register_forward_hook(self.get_llm_activation(name))
            elif "fusion" in name or "merger" in name:
                module.register_forward_hook(self.get_fusion_activation(name))

    def calibrate(self, multimodal_batch):
        """单次前向传播，同时收集所有模态激活"""
        inputs = processor(
            images=multimodal_batch["images"],
            videos=multimodal_batch.get("videos"),
            text=multimodal_batch["text"],
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self.activation_stats

    def compute_quant_params(self):
        """计算所有层的量化参数"""
        vision_params = self.compute_vision_params()  # W8A16
        llm_params = self.compute_llm_params()        # W4A16
        fusion_params = self.compute_fusion_params()  # W8A16

        return {
            "vision": vision_params,
            "llm": llm_params,
            "fusion": fusion_params
        }
```

### 标定数据准备

```python
# 准备多模态标定数据
calibration_data = {
    "image_text": load_coco_captions(num_samples=500),
    "video_text": load_msrvtt(num_samples=200),
    "document_text": load_docvqa(num_samples=300),
}

# 联合标定
calibrator = MultiModalJointCalibrator(model)
for data_type, samples in calibration_data.items():
    for sample in samples:
        calibrator.calibrate(sample)

quant_params = calibrator.compute_quant_params()
```

---

## 标定数据要求

### 数据分布覆盖

| 模态组合 | 数据来源 | 样本数 | 覆盖场景 |
|---------|---------|--------|---------|
| **单图+文本** | COCO Captions | 500 | 通用识别 |
| **多图+文本** | VisualGenome | 300 | 多目标理解 |
| **视频+文本** | MSRVTT | 200 | 视频理解 |
| **文档+文本** | DocVQA | 300 | 文档解析 |
| **长文档+文本** | arXiv papers | 100 | 长上下文 |
| **总计** | - | **1400** | - |

### 数据质量要求

1. **多样性**：覆盖不同场景、光照、分辨率
2. **平衡性**：各模态组合比例合理
3. **代表性**：反映实际部署场景

---

## 验证指标

### 精度对比

| 方法 | MMMU | TextVQA | DocVQA | 说明 |
|------|------|---------|--------|------|
| **ViT 独立标定 + LLM 独立标定** | 56.2% | 62.1% | 78.5% | 基线（用户当前方案） |
| **Vision+LLM 联合标定** | 58.7% | 65.3% | 81.0% | **+2.5% 精度提升** |
| **FP32 基准** | 60.8% | 68.0% | 82.5% | - |

### 关键发现

**联合标定可显著减少多模态量化损失，必须使用！**

---

## 对高通的具体需求

### 标定工具需求

```python
# 期望的 API 设计
qnn.calibrate_multimodal(
    model=model,
    data=multimodal_pairs,
    modalities=["vision", "text", "video"],
    joint_calibration=True,
    fusion_layers=["merger", "cross_attention"],
    num_samples=1000
)
```

### 工具链需求

1. **多模态联合标定 API**
   - 当前状态：无
   - 需求：Vision + LLM 同时标定

2. **跨模态特征对齐验证**
   - 当前状态：无
   - 需求：验证量化后特征对齐质量

3. **自定义数据集接口**
   - 当前状态：有限
   - 需求：用户可上传图文配对数据

---

## 相关文件

- [推理脚本](./inference_script.py)
- [预期输入输出](./expected_input_output.json)
- [标定数据规格](./calibration_data_spec.md)

---

## 参考文档

- [Qwen3-VL 官方任务支持总结](../Qwen3VL_官方任务支持总结.md)
- [Qwen3-VL 困难量化案例指南](../Qwen3VL_困难量化案例_高通芯片部署指南.md)
