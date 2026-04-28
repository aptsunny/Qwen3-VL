"""
Case-04: LoRA + 量化融合推理脚本

本脚本演示 Qwen3-VL 的 LoRA 微调模型量化推理流程，包括：
1. LoRA 权重加载
2. 权重融合
3. 量化推理
4. 多 LoRA 切换

使用方法：
    python inference_script.py --image_path medical_scan.png --prompt "Analyze this image." --lora_path lora_adapters/medical_v1
"""

import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import json
import os


def load_model_with_lora(model_path: str, lora_path: str = None, device: str = "auto"):
    """加载带 LoRA 的 Qwen3-VL 模型"""
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    if lora_path and os.path.exists(lora_path):
        # 加载 LoRA 适配器
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")
    
    return model, processor


def merge_lora_weights(model, lora_alpha: int = 16, lora_r: int = 8):
    """合并 LoRA 权重到基础模型"""
    # 获取 LoRA 参数
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # 计算 ΔW = B × A
            lora_a = module.lora_A.weight
            lora_b = module.lora_B.weight
            delta_w = lora_b @ lora_a
            
            # 合并到基础权重
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight += delta_w * (lora_alpha / lora_r)
            
            # 禁用 LoRA
            module.lora_A = None
            module.lora_B = None
    
    return model


def inference_with_lora(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 512,
    merge_lora: bool = True
):
    """带 LoRA 的推理"""
    # 准备输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # 处理视觉输入
    images, videos = process_vision_info(messages, image_patch_size=16)
    
    # 处理文本
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 准备模型输入
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        return_tensors="pt",
        do_resize=False
    ).to(model.device)
    
    # 推理
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # 解码输出
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 计算统计信息
    total_tokens = outputs.shape[1]
    
    return {
        "response": response,
        "total_tokens": total_tokens,
        "inference_time_ms": inference_time,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
        "lora_merged": merge_lora
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL LoRA + 量化推理脚本")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="模型路径")
    parser.add_argument("--image_path", type=str, required=True, help="图像文件路径")
    parser.add_argument("--prompt", type=str, default="Analyze this image.", help="提示文本")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 适配器路径")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--merge_lora", action="store_true", help="合并 LoRA 权重")
    parser.add_argument("--output_json", type=str, default=None, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model_with_lora(args.model_path, args.lora_path)
    
    # 合并 LoRA 权重（如果指定）
    if args.merge_lora and args.lora_path:
        print("Merging LoRA weights...")
        model = merge_lora_weights(model)
    
    # 推理
    print(f"Processing image: {args.image_path}")
    result = inference_with_lora(
        model=model,
        processor=processor,
        image_path=args.image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        merge_lora=args.merge_lora
    )
    
    # 添加 LoRA 信息
    result["lora_path"] = args.lora_path
    
    # 输出结果
    print("\n=== 推理结果 ===")
    print(f"响应: {result['response'][:200]}...")
    print(f"总 tokens: {result['total_tokens']}")
    print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print(f"峰值内存: {result['peak_memory_mb']:.2f} MB")
    print(f"LoRA 路径: {result['lora_path']}")
    print(f"LoRA 已合并: {result['lora_merged']}")
    
    # 保存结果
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
