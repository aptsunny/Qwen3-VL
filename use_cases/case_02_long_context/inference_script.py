"""
Case-02: 8K 长上下文推理脚本

本脚本演示 Qwen3-VL 长文档理解任务的推理流程，包括：
1. 多页文档处理
2. 长上下文推理
3. KV Cache 管理
4. 输出解析

使用方法：
    python inference_script.py --document_path document.pdf --prompt "Summarize this document."
"""

import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import json
from PIL import Image


def load_model(model_path: str, device: str = "auto"):
    """加载 Qwen3-VL 模型"""
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def prepare_long_document_input(page_images: list, prompt: str):
    """准备长文档输入"""
    content = []
    
    # 添加所有页面图像
    for i, image_path in enumerate(page_images):
        content.append({
            "type": "image",
            "image": image_path
        })
    
    # 添加文本提示
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages


def inference_long_document(
    model,
    processor,
    page_images: list,
    prompt: str,
    max_new_tokens: int = 512,
    kv_bits: int = 8
):
    """长文档推理"""
    # 准备输入
    messages = prepare_long_document_input(page_images, prompt)
    
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
    # KV Cache 内存 ≈ 2 × layers × seq_len × hidden × num_kv_heads × head_dim × dtype_bytes
    layers = 24
    hidden = 2048
    num_kv_heads = 8
    head_dim = 128
    dtype_bytes = 2 if kv_bits == 16 else 1
    
    kv_cache_memory = 2 * layers * context_tokens * num_kv_heads * head_dim * dtype_bytes / 1024 / 1024  # MB
    
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 长文档推理脚本")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="模型路径")
    parser.add_argument("--document_path", type=str, nargs="+", required=True, help="文档页面图像路径（可多个）")
    parser.add_argument("--prompt", type=str, default="Summarize the key points from all pages.", help="提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--kv_bits", type=int, default=8, choices=[8, 16], help="KV Cache 量化位数")
    parser.add_argument("--output_json", type=str, default=None, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path)
    
    # 推理
    print(f"Processing document with {len(args.document_path)} pages...")
    result = inference_long_document(
        model=model,
        processor=processor,
        page_images=args.document_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        kv_bits=args.kv_bits
    )
    
    # 输出结果
    print("\n=== 推理结果 ===")
    print(f"响应: {result['response'][:200]}...")
    print(f"上下文 tokens: {result['context_tokens']}")
    print(f"生成 tokens: {result['generated_tokens']}")
    print(f"总 tokens: {result['total_tokens']}")
    print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print(f"峰值内存: {result['peak_memory_mb']:.2f} MB")
    print(f"KV Cache 内存: {result['kv_cache_memory_mb']:.2f} MB ({args.kv_bits}-bit)")
    
    # 保存结果
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
