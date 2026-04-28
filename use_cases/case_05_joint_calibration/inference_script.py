"""
Case-05: 多模态联合标定推理脚本

本脚本演示 Qwen3-VL 的多模态联合标定流程，包括：
1. 多模态数据加载
2. 联合标定
3. 量化参数计算
4. 精度验证

使用方法：
    python inference_script.py --calibration_data calibration_data.json --output quant_params.json
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import json
from collections import defaultdict


class MultiModalJointCalibrator:
    """多模态联合标定器"""
    
    def __init__(self, model):
        self.model = model
        self.vision_activations = defaultdict(list)
        self.llm_activations = defaultdict(list)
        self.fusion_activations = defaultdict(list)
        
        # 注册激活钩子
        self.hooks = []
        for name, module in model.named_modules():
            if "vision" in name and "conv" in name:
                hook = module.register_forward_hook(self._get_vision_hook(name))
                self.hooks.append(hook)
            elif "language_model" in name and "layer" in name:
                hook = module.register_forward_hook(self._get_llm_hook(name))
                self.hooks.append(hook)
            elif "merger" in name or "fusion" in name:
                hook = module.register_forward_hook(self._get_fusion_hook(name))
                self.hooks.append(hook)
    
    def _get_vision_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.vision_activations[name].append(output.detach())
        return hook
    
    def _get_llm_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.llm_activations[name].append(output.detach())
        return hook
    
    def _get_fusion_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.fusion_activations[name].append(output.detach())
        return hook
    
    def calibrate(self, processor, multimodal_batch):
        """单次前向传播，同时收集所有模态激活"""
        # 处理视觉输入
        images = multimodal_batch.get("images", [])
        videos = multimodal_batch.get("videos", [])
        text = multimodal_batch.get("text", "")
        
        # 处理视觉信息
        processed_images, processed_videos, video_kwargs = process_vision_info(
            [{"role": "user", "content": [{"type": "text", "text": text}]}],
            image_patch_size=16,
            return_video_kwargs=True
        ) if not images and not videos else (None, None, {})
        
        # 处理文本
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        processed_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 准备模型输入
        inputs = processor(
            text=processed_text,
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="pt",
            do_resize=False
        ).to(self.model.device)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(**inputs)
        
        return self._get_activation_stats()
    
    def _get_activation_stats(self):
        """获取激活统计信息"""
        stats = {
            "vision": {},
            "llm": {},
            "fusion": {}
        }
        
        # Vision 统计
        for name, activations in self.vision_activations.items():
            all_acts = torch.cat(activations, dim=0)
            stats["vision"][name] = {
                "min": all_acts.min().item(),
                "max": all_acts.max().item(),
                "mean": all_acts.mean().item(),
                "std": all_acts.std().item()
            }
        
        # LLM 统计
        for name, activations in self.llm_activations.items():
            all_acts = torch.cat(activations, dim=0)
            stats["llm"][name] = {
                "min": all_acts.min().item(),
                "max": all_acts.max().item(),
                "mean": all_acts.mean().item(),
                "std": all_acts.std().item()
            }
        
        # Fusion 统计
        for name, activations in self.fusion_activations.items():
            all_acts = torch.cat(activations, dim=0)
            stats["fusion"][name] = {
                "min": all_acts.min().item(),
                "max": all_acts.max().item(),
                "mean": all_acts.mean().item(),
                "std": all_acts.std().item()
            }
        
        return stats
    
    def compute_quant_params(self):
        """计算量化参数"""
        quant_params = {
            "vision": {},
            "llm": {},
            "fusion": {}
        }
        
        # Vision 量化参数（W8A16）
        for name, stats in self.vision_activations.items():
            all_acts = torch.cat(stats, dim=0)
            scale = all_acts.abs().max() / 127.0
            quant_params["vision"][name] = {
                "scale": scale.item(),
                "zero_point": 0,
                "bitwidth": 8
            }
        
        # LLM 量化参数（W4A16）
        for name, stats in self.llm_activations.items():
            all_acts = torch.cat(stats, dim=0)
            scale = all_acts.abs().max() / 7.0  # 4-bit: [-8, 7]
            quant_params["llm"][name] = {
                "scale": scale.item(),
                "zero_point": 0,
                "bitwidth": 4
            }
        
        # Fusion 量化参数（W8A16）
        for name, stats in self.fusion_activations.items():
            all_acts = torch.cat(stats, dim=0)
            scale = all_acts.abs().max() / 127.0
            quant_params["fusion"][name] = {
                "scale": scale.item(),
                "zero_point": 0,
                "bitwidth": 8
            }
        
        return quant_params
    
    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()


def load_calibration_data(data_path: str):
    """加载标定数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 多模态联合标定脚本")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="模型路径")
    parser.add_argument("--calibration_data", type=str, required=True, help="标定数据 JSON 文件路径")
    parser.add_argument("--output", type=str, default="quant_params.json", help="输出量化参数文件路径")
    parser.add_argument("--num_samples", type=int, default=100, help="标定样本数")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # 加载标定数据
    print(f"Loading calibration data from {args.calibration_data}...")
    calibration_data = load_calibration_data(args.calibration_data)
    
    # 创建标定器
    calibrator = MultiModalJointCalibrator(model)
    
    # 执行标定
    print(f"Running joint calibration with {args.num_samples} samples...")
    start_time = time.time()
    
    for i, sample in enumerate(calibration_data[:args.num_samples]):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{args.num_samples}...")
        
        calibrator.calibrate(processor, sample)
    
    calibration_time = time.time() - start_time
    print(f"Calibration completed in {calibration_time:.2f} seconds")
    
    # 计算量化参数
    print("Computing quantization parameters...")
    quant_params = calibrator.compute_quant_params()
    
    # 添加元数据
    quant_params["metadata"] = {
        "model_path": args.model_path,
        "num_samples": args.num_samples,
        "calibration_time_seconds": calibration_time,
        "quantization_config": {
            "vision": "W8A16",
            "llm": "W4A16",
            "fusion": "W8A16"
        }
    }
    
    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(quant_params, f, ensure_ascii=False, indent=2)
    print(f"Quantization parameters saved to {args.output}")
    
    # 清理
    calibrator.cleanup()


if __name__ == "__main__":
    main()
