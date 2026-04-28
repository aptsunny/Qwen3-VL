"""
Case-01: 视频输入量化推理脚本

本脚本演示 Qwen3-VL 视频理解任务的推理流程，包括：
1. 视频帧采样
2. 多模态输入处理
3. 模型推理
4. 输出解析

使用方法：
    python inference_script.py --video_path sample_video.mp4 --prompt "Describe this video."
"""

import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import json


def load_model(model_path: str, device: str = "auto"):
    """加载 Qwen3-VL 模型"""
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def prepare_video_input(video_path: str, prompt: str, fps: float = 2.0, max_frames: int = 16):
    """准备视频输入"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "max_pixels": 256 * 32 * 32,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages


def inference_video(
    model,
    processor,
    video_path: str,
    prompt: str,
    fps: float = 2.0,
    max_frames: int = 16,
    max_new_tokens: int = 512
):
    """视频推理"""
    # 准备输入
    messages = prepare_video_input(video_path, prompt, fps, max_frames)
    
    # 处理视觉输入
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    if videos:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    
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
        video_metadata=video_metadatas if videos else None,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs
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
    video_tokens = inputs["input_ids"].shape[1]
    
    return {
        "response": response,
        "video_tokens": video_tokens,
        "total_tokens": total_tokens,
        "inference_time_ms": inference_time,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 视频推理脚本")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="模型路径")
    parser.add_argument("--video_path", type=str, required=True, help="视频文件路径")
    parser.add_argument("--prompt", type=str, default="Describe what happens in this video.", help="提示文本")
    parser.add_argument("--fps", type=float, default=2.0, help="视频帧采样率")
    parser.add_argument("--max_frames", type=int, default=16, help="最大帧数")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--output_json", type=str, default=None, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path)
    
    # 推理
    print(f"Processing video: {args.video_path}")
    result = inference_video(
        model=model,
        processor=processor,
        video_path=args.video_path,
        prompt=args.prompt,
        fps=args.fps,
        max_frames=args.max_frames,
        max_new_tokens=args.max_new_tokens
    )
    
    # 输出结果
    print("\n=== 推理结果 ===")
    print(f"响应: {result['response']}")
    print(f"视频 tokens: {result['video_tokens']}")
    print(f"总 tokens: {result['total_tokens']}")
    print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print(f"峰值内存: {result['peak_memory_mb']:.2f} MB")
    
    # 保存结果
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
