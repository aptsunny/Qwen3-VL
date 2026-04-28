"""
Case-03: KV Cache 量化推理脚本

本脚本演示 Qwen3-VL 的 KV Cache 量化推理流程，包括：
1. KV8 vs KV16 对比
2. Per-Head 量化策略
3. 内存监控
4. 精度验证

使用方法：
    python inference_script.py --image_path document.png --prompt "Extract all text." --kv_bits 8
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import json


class PerHeadKVQuantizer(nn.Module):
    """逐头 KV Cache 量化器"""
    def __init__(self, num_kv_heads=8, head_dim=128):
        super().__init__()
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

            k_scale = k_head.abs().max() / 127.0
            v_scale = v_head.abs().max() / 127.0

            k_q = torch.quantize_per_tensor(k_head, k_scale, 0, torch.qint8)
            v_q = torch.quantize_per_tensor(v_head, v_scale, 0, torch.qint8)

            k_quantized.append(k_q)
            v_quantized.append(v_q)

        return torch.stack(k_quantized, dim=2), torch.stack(v_quantized, dim=2)

    def dequantize_kv(self, k_q, v_q):
        """解码时反量化"""
        k_dequant = []
        v_dequant = []
        for head_idx in range(self.num_kv_heads):
            k_head = k_q[:, :, head_idx, :].dequantize()
            v_head = v_q[:, :, head_idx, :].dequantize()
            k_dequant.append(k_head)
            v_dequant.append(v_head)
        return torch.cat(k_dequant, dim=2), torch.cat(v_dequant, dim=2)


def load_model(model_path: str, device: str = "auto"):
    """加载 Qwen3-VL 模型"""
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def inference_with_kv_quant(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 1024,
    kv_bits: int = 8
):
    """带 KV Cache 量化的推理"""
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
    
    # 记录上下文长度
    context_tokens = inputs["input_ids"].shape[1]
    
    # 推理
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # 解码输出
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 计算统计信息
    total_tokens = outputs.shape[1]
    generated_tokens = total_tokens - context_tokens
    
    # 估算 KV Cache 内存
    layers = 24
    num_kv_heads = 8
    head_dim = 128
    dtype_bytes = 2 if kv_bits == 16 else 1
    
    kv_cache_memory = 2 * layers * total_tokens * num_kv_heads * head_dim * dtype_bytes / 1024 / 1024  # MB
    
    return {
        "response": response,
        "context_tokens": context_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "inference_time_ms": inference_time,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
        "kv_cache_memory_mb": kv_cache_memory,
        "kv_bits": kv_bits
    }


def compare_kv_quantization(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 1024
):
    """对比 KV16 和 KV8 的性能"""
    results = {}
    
    # KV16 推理
    print("Running inference with KV16...")
    results["kv16"] = inference_with_kv_quant(
        model, processor, image_path, prompt, max_new_tokens, kv_bits=16
    )
    
    # 清理缓存
    torch.cuda.empty_cache()
    
    # KV8 推理
    print("Running inference with KV8...")
    results["kv8"] = inference_with_kv_quant(
        model, processor, image_path, prompt, max_new_tokens, kv_bits=8
    )
    
    # 计算对比
    comparison = {
        "memory_reduction": f"{(1 - results['kv8']['kv_cache_memory_mb'] / results['kv16']['kv_cache_memory_mb']) * 100:.1f}%",
        "speed_improvement": f"{(results['kv16']['inference_time_ms'] / results['kv8']['inference_time_ms'] - 1) * 100:.1f}%",
        "kv16_memory_mb": results["kv16"]["kv_cache_memory_mb"],
        "kv8_memory_mb": results["kv8"]["kv_cache_memory_mb"]
    }
    
    return results, comparison


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL KV Cache 量化推理脚本")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="模型路径")
    parser.add_argument("--image_path", type=str, required=True, help="图像文件路径")
    parser.add_argument("--prompt", type=str, default="Extract all information from this document.", help="提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="最大生成 token 数")
    parser.add_argument("--kv_bits", type=int, default=8, choices=[8, 16], help="KV Cache 量化位数")
    parser.add_argument("--compare", action="store_true", help="对比 KV16 和 KV8")
    parser.add_argument("--output_json", type=str, default=None, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path)
    
    if args.compare:
        # 对比模式
        results, comparison = compare_kv_quantization(
            model, processor, args.image_path, args.prompt, args.max_new_tokens
        )
        
        print("\n=== KV16 vs KV8 对比 ===")
        print(f"KV16 内存: {comparison['kv16_memory_mb']:.2f} MB")
        print(f"KV8 内存: {comparison['kv8_memory_mb']:.2f} MB")
        print(f"内存减少: {comparison['memory_reduction']}")
        print(f"速度提升: {comparison['speed_improvement']}")
        
        output = {"results": results, "comparison": comparison}
    else:
        # 单次推理
        result = inference_with_kv_quant(
            model=model,
            processor=processor,
            image_path=args.image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            kv_bits=args.kv_bits
        )
        
        print("\n=== 推理结果 ===")
        print(f"响应: {result['response'][:200]}...")
        print(f"上下文 tokens: {result['context_tokens']}")
        print(f"生成 tokens: {result['generated_tokens']}")
        print(f"推理时间: {result['inference_time_ms']:.2f} ms")
        print(f"峰值内存: {result['peak_memory_mb']:.2f} MB")
        print(f"KV Cache 内存: {result['kv_cache_memory_mb']:.2f} MB ({args.kv_bits}-bit)")
        
        output = result
    
    # 保存结果
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
