# Case-01: 视频输入量化

## 案例概述

**难度等级**：⭐⭐⭐⭐⭐（最高难度）

**任务类型**：Video Understanding（视频理解）

**推荐模型**：Qwen3-VL-2B-Instruct / Qwen3-VL-4B-Instruct

**核心挑战**：视频输入的多模态量化是 Qwen3-VL 部署中最具挑战性的场景，涉及时序一致性、内存爆炸、变长序列等多个难题。

---

## 为什么这个案例难量化？

### 难点 1：时序一致性

**问题描述**：
视频帧之间存在时序相关性，独立量化每帧会导致时序特征不一致。

**量化影响**：
- 运动轨迹识别错误
- 动作识别精度下降 5-8%
- 时序注意力权重失真

**根本原因**：
```python
# 传统量化方式（错误）
for frame in video_frames:
    frame_features = vision_encoder(frame)  # 每帧独立量化
    # 问题：帧间量化参数不同，导致特征分布不一致

# 正确方式（联合量化）
video_features = vision_encoder(video_frames)  # 整体量化，保持时序一致性
```

### 难点 2：内存爆炸

**问题描述**：
视频推理的内存消耗 = 单帧内存 × 帧数，线性增长。

**内存计算**：
```
输入：640×480 视频，16 帧，fps=2
单帧 patch 数 = (640/16) × (480/16) = 1200
单帧视觉 token = 1200 × 32² = 1,228,800
单帧视觉特征内存 = 1200 × 1280 × 2 bytes ≈ 3 MB
视频总视觉内存 = 3 MB × 16 × 2 = 96 MB（仅视觉编码器）

LLM 解码器内存（峰值）：
- KV Cache：seq_len=8192, hidden=2048, layers=24, KV heads=16
- KV Cache 内存 = 8192 × 2048 × 2 × 24 × 2 × 2 bytes ≈ 3.2 GB（KV16）

总峰值内存 ≈ 3.3 GB（2B 模型，16 帧 480p）
```

**移动端限制**：
- 骁龙平台可用内存：~4-6 GB（含系统占用）
- 建议视频配置：**8 帧，360p（640×360）**
- 峰值内存控制：≤ 2 GB

### 难点 3：变长序列量化

**问题描述**：
视频帧数不固定（8-2048 帧），量化参数需适应不同序列长度。

**传统 PTQ 的缺陷**：
- 标定时用固定长度（如 16 帧）
- 部署时遇到更长序列 → 激活分布偏移 → 精度下降

**示例**：
```python
# 标定时
calibration_video = load_video("video.mp4", num_frames=16)
quant_params = calibrate(model, calibration_video)

# 部署时（问题）
deploy_video = load_video("long_video.mp4", num_frames=64)  # 更长序列
# 激活分布偏移，量化参数不再适用
output = quantized_model(deploy_video)  # 精度下降
```

### 难点 4：KV Cache 压力

**问题描述**：
视频理解需要处理长序列（40K-100K tokens），KV Cache 成为内存瓶颈。

**KV Cache 内存计算**：
```
Qwen3-VL-2B:
- layers = 24
- hidden_size = 2048
- num_kv_heads = 8（GQA）

KV16（FP16）内存：
单层 KV = 2 × seq_len × 8 × 128 × 2 bytes = 4096 × seq_len bytes
24 层总内存 = 98,304 × seq_len bytes

示例：
seq_len = 40,960（视频 16 帧）→ 98,304 × 40960 ≈ 3.9 GB
seq_len = 81,920（视频 32 帧）→ 7.8 GB
```

---

## 预期输入输出

### 输入规格

```json
{
  "input_type": "video",
  "video_path": "sample_video.mp4",
  "video_config": {
    "fps": 2.0,
    "num_frames": 16,
    "min_pixels": 256 * 32 * 32,
    "max_pixels": 256 * 32 * 32,
    "total_pixels": 16384 * 32 * 32
  },
  "text_prompt": "Describe what happens in this video in detail.",
  "max_new_tokens": 512
}
```

### 输出规格

```json
{
  "output_type": "text",
  "response": "The video shows a person walking through a park on a sunny day. They pass by several trees and a small pond where ducks are swimming. The person stops briefly to watch the ducks before continuing their walk.",
  "video_tokens": 38400,
  "total_tokens": 39500,
  "inference_time_ms": 1500,
  "peak_memory_mb": 3300
}
```

---

## 量化方案建议

### 推荐配置

| 模块 | 量化方案 | 精度损失 | 内存减少 | 推荐 |
|------|---------|---------|---------|------|
| **Vision Encoder** | W8A8 | -2.4% | 2× | ⚠️ 不推荐 |
| **Vision Encoder** | W8A16 | -0.7% | 1.5× | ✅ **推荐** |
| **LLM Weights** | W4A16 | -1.3% | 2.2× | ✅ **推荐** |
| **KV Cache** | KV8 | -0.3% | 2× | ✅ **必须** |

### 最佳组合

```
Vision W8A16 + LLM W4A16 + KV8
- 总精度损失：~1.2%
- 总内存减少：~3.5×
- 移动端可行
```

---

## 标定数据要求

### 数据分布覆盖

| 维度 | 类别数 | 每类样本数 | 总样本数 |
|------|-------|-----------|---------|
| **动作类型** | 20+ | 5 视频/类 | 100+ 视频 |
| **场景** | 10+（室内/外/街景） | 10 视频/场景 | 100+ 视频 |
| **光照条件** | 5（亮/暗/逆光/夜间/模糊） | 20 视频/条件 | 100 视频 |
| **帧率** | 3（1fps/2fps/4fps） | 30 视频/帧率 | 90 视频 |
| **分辨率** | 3（360p/480p/720p） | 30 视频/分辨率 | 90 视频 |
| **总计** | - | - | **≥ 500 视频片段** |

### 标定流程

```python
# Step 1: 准备标定数据
calibration_videos = [
    ("video1.mp4", 16, "action recognition"),
    ("video2.mp4", 8, "object tracking"),
    # ... 覆盖主要分布
]

# Step 2: 联合标定（Vision + LLM 同时）
class VideoJointCalibrator:
    def __init__(self, model):
        self.model = model
        self.vision_activations = {}
        self.llm_activations = {}

    def calibrate(self, video_batch, text_prompts):
        """单次前向传播，同时收集 Vision 和 LLM 激活"""
        inputs = processor(
            videos=video_batch,
            text=text_prompts,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        return self.activation_stats

# Step 3: 应用量化并验证
calibrator = VideoJointCalibrator(model)
for video, prompt in calibration_videos:
    frames = sample_frames(video, num_frames=16)
    calibrator.calibrate(frames, prompt)

quant_params = calibrator.compute_quant_params()
quantized_model = apply_quantization(model, quant_params)
```

---

## 验证指标

### 精度验证

| 数据集 | 指标 | FP16 基准 | 量化目标 | 可接受损失 |
|-------|------|----------|---------|-----------|
| **MSRVTT** | CIDEr | 65.2 | ≥ 63.0 | ≤ 2.2 |
| **TGIF** | Action Acc | 78.5% | ≥ 75.0% | ≤ 3.5% |
| **ActivityNet** | mAP | 35.2% | ≥ 33.0% | ≤ 2.2% |

### 性能验证

| 平台 | 帧数 | 分辨率 | 目标延迟 | 目标内存 |
|------|------|--------|---------|---------|
| 骁龙 8 Gen 3 | 8 帧 | 360p | ≤ 1.5 s | ≤ 2 GB |
| 骁龙 8 Gen 3 | 16 帧 | 480p | ≤ 2.5 s | ≤ 3.5 GB |

---

## 对高通的具体需求

### QNN 工具链需求

1. **视频 3D 卷积 INT8 量化支持**
   - 当前状态：未知
   - 需求：支持视频帧批处理的 INT8 量化

2. **多帧批处理并行化**
   - 当前状态：部分支持
   - 需求：8-16 帧的并行推理优化

3. **动态 shape 编译**
   - 当前状态：部分支持
   - 需求：完全动态 batch/seq_len 支持

4. **Per-Head KV 量化**
   - 当前状态：可能不支持
   - 需求：逐头 KV Cache 量化策略

### 标定工具需求

```python
# 期望的 API 设计
qnn.calibrate_video(
    model=model,
    video_data=video_text_pairs,
    modalities=["vision", "text", "video"],
    joint_calibration=True,
    kv_quantization="per_head",
    frame_lengths=[8, 16, 32]
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
- [Video Understanding Cookbook](../cookbooks/video_understanding.ipynb)
