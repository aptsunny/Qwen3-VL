"""
Qwen3-VL USE Case 综合测试脚本

本脚本运行所有 USE Case 并记录输入输出，生成总结表格。

使用方法：
    rlaunch --cpu 32 --gpu 1 --memory 80896 --charged-group on_device --private-machine group -- python run_all_use_cases.py
"""

import argparse
import torch
import json
import time
import os
from datetime import datetime
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import sys


class UseCaseRunner:
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.results = []
    
    def load_model(self):
        print(f"\n{'='*60}")
        print(f"Loading model: {self.model_name} from {self.model_path}")
        print(f"{'='*60}")
        
        start_time = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        load_time = time.time() - start_time
        
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Device: {self.model.device}")
        return load_time
    
    def run_single_image_inference(self, image_path: str, prompt: str, case_name: str, max_new_tokens: int = 256):
        """单图推理"""
        print(f"\n--- Running {case_name} ({self.model_name}) ---")
        print(f"Image: {image_path}")
        print(f"Prompt: {prompt}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        images, videos = process_vision_info(messages, image_patch_size=16)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False).to(self.model.device)
        
        input_tokens = inputs["input_ids"].shape[1]
        
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        output_tokens = outputs.shape[1]
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats()
        
        result = {
            "case_name": case_name,
            "model": self.model_name,
            "input_type": "single_image",
            "image_path": image_path,
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generated_tokens": output_tokens - input_tokens,
            "response": response,
            "inference_time_ms": round(inference_time * 1000, 2),
            "peak_memory_mb": round(peak_memory, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result)
        print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"Inference time: {inference_time*1000:.2f} ms, Peak memory: {peak_memory:.2f} MB")
        print(f"Response: {response[:300]}...")
        
        return result
    
    def run_multi_image_inference(self, image_paths: list, prompt: str, case_name: str, max_new_tokens: int = 512):
        """多图推理"""
        print(f"\n--- Running {case_name} ({self.model_name}) ---")
        print(f"Images: {image_paths}")
        print(f"Prompt: {prompt}")
        
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        images, videos = process_vision_info(messages, image_patch_size=16)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False).to(self.model.device)
        
        input_tokens = inputs["input_ids"].shape[1]
        
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        output_tokens = outputs.shape[1]
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats()
        
        result = {
            "case_name": case_name,
            "model": self.model_name,
            "input_type": "multi_image",
            "image_paths": image_paths,
            "num_images": len(image_paths),
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generated_tokens": output_tokens - input_tokens,
            "response": response,
            "inference_time_ms": round(inference_time * 1000, 2),
            "peak_memory_mb": round(peak_memory, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result)
        print(f"Num images: {len(image_paths)}, Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"Inference time: {inference_time*1000:.2f} ms, Peak memory: {peak_memory:.2f} MB")
        print(f"Response: {response[:300]}...")
        
        return result
    
    def run_2d_grounding(self, image_path: str, prompt: str, case_name: str):
        """2D 目标定位"""
        print(f"\n--- Running {case_name} ({self.model_name}) ---")
        print(f"Image: {image_path}")
        print(f"Prompt: {prompt}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        images, videos = process_vision_info(messages, image_patch_size=16)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, videos=videos, return_tensors="pt", do_resize=False).to(self.model.device)
        
        input_tokens = inputs["input_ids"].shape[1]
        
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        output_tokens = outputs.shape[1]
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats()
        
        result = {
            "case_name": case_name,
            "model": self.model_name,
            "input_type": "2d_grounding",
            "image_path": image_path,
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generated_tokens": output_tokens - input_tokens,
            "response": response,
            "inference_time_ms": round(inference_time * 1000, 2),
            "peak_memory_mb": round(peak_memory, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(result)
        print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"Inference time: {inference_time*1000:.2f} ms, Peak memory: {peak_memory:.2f} MB")
        print(f"Response: {response}")
        
        return result
    
    def run_ocr(self, image_path: str, prompt: str, case_name: str):
        """OCR 任务"""
        return self.run_single_image_inference(image_path, prompt, case_name, max_new_tokens=512)
    
    def run_document_parsing(self, image_path: str, prompt: str, case_name: str):
        """文档解析"""
        return self.run_single_image_inference(image_path, prompt, case_name, max_new_tokens=1024)
    
    def save_results(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL USE Case 综合测试")
    parser.add_argument("--model_2b", type=str, default="/data/models/Qwen3-VL-2B-Instruct", help="2B 模型路径")
    parser.add_argument("--model_4b", type=str, default="/data/models/Qwen3-VL-4B-Instruct", help="4B 模型路径")
    parser.add_argument("--output_dir", type=str, default="/data/sunyue_projects/Qwen3-VL-Demo-Project/use_cases/results", help="输出目录")
    parser.add_argument("--model_only", type=str, default=None, choices=["2b", "4b"], help="只测试指定模型")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    test_cases = [
        {
            "name": "Case_01_OCR",
            "image": "/data/sunyue_projects/Qwen3-VL-Demo-Project/cookbooks/assets/ocr/ocr_example1.jpg",
            "prompt": "Extract all text from this image.",
            "type": "ocr"
        },
        {
            "name": "Case_02_OCR_Chinese",
            "image": "/data/sunyue_projects/Qwen3-VL-Demo-Project/cookbooks/assets/ocr/ocr_example3.jpg",
            "prompt": "识别图片中的所有文字。",
            "type": "ocr"
        },
        {
            "name": "Case_03_Omni_Recognition",
            "image": "/data/sunyue_projects/Qwen3-VL-Demo-Project/cookbooks/assets/omni_recognition/sample-food.jpeg",
            "prompt": "Identify all objects in this image.",
            "type": "single_image"
        },
        {
            "name": "Case_04_2D_Grounding",
            "image": "/data/sunyue_projects/Qwen3-VL-Demo-Project/cookbooks/assets/spatial_understanding/dining_table.png",
            "prompt": "Locate all objects on the table and provide their bounding boxes in JSON format with relative coordinates (0-1000).",
            "type": "2d_grounding"
        },
        {
            "name": "Case_05_Document_Parsing",
            "image": "/data/sunyue_projects/Qwen3-VL-Demo-Project/cookbooks/assets/document_parsing/docparsing_example1.jpg",
            "prompt": "Parse this document and output the structure in JSON format.",
            "type": "document_parsing"
        },
        {
            "name": "Case_06_Spatial_Understanding",
            "image": "/data/sunyue_projects/Qwen3-VL-Demo-Project/cookbooks/assets/spatial_understanding/office.jpg",
            "prompt": "Describe the spatial layout of this office. Where are the main objects located?",
            "type": "single_image"
        },
        {
            "name": "Case_07_Caption",
            "image": "/data/sunyue_projects/Qwen3-VL-Demo-Project/cookbooks/assets/omni_recognition/sample-bird.jpg",
            "prompt": "Describe this image in detail.",
            "type": "single_image"
        }
    ]
    
    models_to_test = []
    if args.model_only == "2b":
        models_to_test = [(args.model_2b, "Qwen3-VL-2B")]
    elif args.model_only == "4b":
        models_to_test = [(args.model_4b, "Qwen3-VL-4B")]
    else:
        models_to_test = [(args.model_2b, "Qwen3-VL-2B"), (args.model_4b, "Qwen3-VL-4B")]
    
    for model_path, model_name in models_to_test:
        print(f"\n{'#'*80}")
        print(f"# Testing Model: {model_name}")
        print(f"# Path: {model_path}")
        print(f"{'#'*80}")
        
        if not os.path.exists(model_path):
            print(f"Model path not found: {model_path}, skipping...")
            continue
        
        runner = UseCaseRunner(model_path, model_name)
        runner.load_model()
        
        for case in test_cases:
            if not os.path.exists(case["image"]):
                print(f"Image not found: {case['image']}, skipping...")
                continue
            
            try:
                if case["type"] == "ocr":
                    runner.run_ocr(case["image"], case["prompt"], case["name"])
                elif case["type"] == "2d_grounding":
                    runner.run_2d_grounding(case["image"], case["prompt"], case["name"])
                elif case["type"] == "document_parsing":
                    runner.run_document_parsing(case["image"], case["prompt"], case["name"])
                else:
                    runner.run_single_image_inference(case["image"], case["prompt"], case["name"])
            except Exception as e:
                print(f"Error running {case['name']}: {e}")
                import traceback
                traceback.print_exc()
                runner.results.append({
                    "case_name": case["name"],
                    "model": model_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        output_file = os.path.join(args.output_dir, f"{model_name}_results.json")
        runner.save_results(output_file)
        all_results[model_name] = runner.results
        
        del runner.model
        torch.cuda.empty_cache()
    
    summary_file = os.path.join(args.output_dir, "all_results_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\nAll results summary saved to {summary_file}")


if __name__ == "__main__":
    main()
