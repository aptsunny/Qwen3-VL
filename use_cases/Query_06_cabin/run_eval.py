#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query_06_cabin 推理 + 评测一体脚本（单模型跑通；自包含，除图片外无外部依赖）。

读 data/eval.jsonl（build_dataset.py 产出的 100 条：闲聊 / VQA车内 / VQA车外 / ToolCall），
用本地 Qwen3-VL 权重做推理，随后打分，写 results/<tag>/{outputs,scores}.jsonl + summary.json。

评分口径对齐上游车载多任务评测基准（同 judge 模型、同 P_REF prompt、同 toolcall 规则），
相关逻辑内联于本文件：
  - ToolCall  → 程序判分（工具名命中 + 参数语义命中），无需 API key；
  - VQA/闲聊  → gpt-oss-120b judge（models-proxy），需 MODELS_PROXY_API_KEY（只经环境变量传入）。
                未提供 key 时这两类标记 verdict=skip、不报错退出。

用法：
  # 推理 + 评分（judge 需 key；toolcall 不需要）
  MODELS_PROXY_API_KEY=<key> python3 run_eval.py --model /data/models/Qwen3-VL-2B-Instruct --tag qwen3vl-2b
  MODELS_PROXY_API_KEY=<key> python3 run_eval.py --model /data/models/Qwen3-VL-4B-Instruct --tag qwen3vl-4b
  python3 run_eval.py --model <hf> --limit 8            # 冒烟：前 8 条
  python3 run_eval.py --compare                         # 汇总 results/*/summary.json 成对比表
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

JUDGE_URL = os.environ.get("JUDGE_URL", "https://api.openai.com/v1/chat/completions")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-oss-120b")
KEY = os.environ.get("MODELS_PROXY_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


# ============================ 推理段 ============================
# 精简自上游基准的推理脚本 + tau2/server.py（内联，不依赖外部包）。

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.I)
_THINK_CLOSE = "</think>"


def _decode_json(value: str):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _calls_from_json(value) -> list[dict]:
    """常见 Qwen JSON 形态 → OpenAI function-call 列表。"""
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        candidates = value["tool_calls"]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = [value]
    calls = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        fn = cand.get("function")
        if not isinstance(fn, dict):
            fn = cand
        name = fn.get("name") or cand.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = fn.get("arguments", cand.get("arguments", {}))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                      "function": {"name": name, "arguments": args}})
    return calls


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Qwen XML/JSON tool-call 输出 → (content, tool_calls)；解析不出则原样留 content。"""
    text = text or ""
    matches = list(_TOOL_CALL_TAG.finditer(text))
    if matches:
        calls = []
        for m in matches:
            calls.extend(_calls_from_json(_decode_json(m.group(1).strip())))
        if calls:
            return _TOOL_CALL_TAG.sub("", text).strip(), calls
    calls = _calls_from_json(_decode_json(text.strip()))
    if calls:
        return "", calls
    return text, []


def parse_assistant_output(text: str) -> tuple[str, list[dict]]:
    """剥离 <think>…</think> 推理段，再解析 tool_calls。"""
    text = text or ""
    end = text.find(_THINK_CLOSE)
    if end >= 0:
        text = text[end + len(_THINK_CLOSE):].lstrip("\n")
    content, calls = parse_tool_calls(text)
    return (content.strip() if calls else content), calls


def prepare_messages(sample: dict) -> tuple[list[dict], list[str]]:
    """把 image_url part 换成 {"type":"image"} 标记，解析出本地图片路径（相对 data/images/）。"""
    messages, image_paths = [], []
    for msg in sample["messages"]:
        content = msg.get("content")
        if not isinstance(content, list):
            messages.append(msg)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                name = (url[7:] if url.startswith("file://") else url)
                path = DATA_DIR / "images" / Path(name).name
                image_paths.append(str(path))
                parts.append({"type": "image"})
            else:
                parts.append(part)
        messages.append({"role": msg["role"], "content": parts})
    return messages, image_paths


def run_inference(model_path: str, samples: list[dict], out_path: Path,
                  dtype: str = "bf16", enable_thinking: bool = False) -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    print(f"[load] {model_path}  dtype={dtype}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch_dtype, device_map="cuda")
    model.eval()
    device = next(model.parameters()).device

    stats = {"ok": 0, "fail": 0}
    with out_path.open("w", encoding="utf-8") as f:
        for idx, s in enumerate(samples, 1):
            try:
                messages, image_paths = prepare_messages(s)
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    **({"tools": s["tools"]} if s.get("tools") else {}))
                pk = {"text": [prompt], "padding": True, "return_tensors": "pt"}
                if image_paths:
                    pk["images"] = [Image.open(p).convert("RGB") for p in image_paths]
                gp = s.get("gen_params") or {}
                max_new = min(int(gp.get("max_tokens") or 512), 2048)
                started = time.time()
                with torch.inference_mode():
                    inputs = processor(**pk).to(device)
                    gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                         repetition_penalty=1.0)
                n_prompt = int(inputs.input_ids.shape[1])
                comp = gen[:, n_prompt:]
                text = processor.batch_decode(comp, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
                content, tool_calls = parse_assistant_output(text)
                rec = {"sample_id": s["sample_id"], "category": s["category"],
                       "content": content, "tool_calls": tool_calls or None,
                       "usage": {"prompt_tokens": n_prompt, "completion_tokens": int(comp.shape[1])},
                       "latency_s": round(time.time() - started, 2)}
                stats["ok"] += 1
            except Exception as e:
                rec = {"sample_id": s["sample_id"], "category": s["category"],
                       "content": None, "tool_calls": None, "error": str(e)[:300]}
                stats["fail"] += 1
                print(f"  FAIL {s['sample_id']}: {str(e)[:160]}", flush=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if idx % 10 == 0 or idx == len(samples):
                print(f"  infer {idx}/{len(samples)} ok={stats['ok']} fail={stats['fail']}", flush=True)
    print(f"[infer] done ok={stats['ok']} fail={stats['fail']} -> {out_path}", flush=True)


# ============================ 评分段：ToolCall 程序判分 ============================
# 内联自上游基准 score.py:judge_toolcall / _semantic_query_ok / _norm_calls（口径逐条对齐）。

_AMBIGUOUS_INPUTS = {"调成Low", "调成High"}
_TEMP_EQ = {"low": {"low", "16"}, "16": {"low", "16"}, "high": {"high", "28"}, "28": {"high", "28"}}


def _semantic_query_ok(want, got) -> bool:
    norm = lambda s: re.sub(r"[^\w一-鿿]", "", (s or "").lower())
    a, b = norm(want), norm(got)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    ga = {a[i:i + 2] for i in range(max(1, len(a) - 1))}
    gb = {b[i:i + 2] for i in range(max(1, len(b) - 1))}
    return len(ga & gb) / max(1, len(ga)) >= 0.5


def _norm_calls(tool_calls) -> list[dict]:
    out = []
    for c in tool_calls or []:
        fn = c.get("function") if isinstance(c, dict) else None
        if not isinstance(fn, dict):
            fn = c if isinstance(c, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        out.append({"name": name,
                    "arguments": {k: str(v).strip('"').strip("'") for k, v in args.items()}})
    return out


def judge_toolcall(expect: dict, content: str, tool_calls, user_query: str) -> dict:
    exp_name, exp_params = expect["name"], expect.get("params") or {}
    calls = _norm_calls(tool_calls)
    got_names = [c["name"] for c in calls]
    name_hit = exp_name in got_names
    param_hit = param_hit_exact = None
    if exp_params and name_hit:
        got = next(c for c in calls if c["name"] == exp_name)["arguments"]
        if str(user_query or "").strip() in _AMBIGUOUS_INPUTS:
            param_hit = param_hit_exact = None
        else:
            exact_ok = sem_ok = True
            for k, want in exp_params.items():
                have = got.get(k, "")
                if have.lower() != str(want).lower():
                    exact_ok = False
                if k == "sound_position":            # 音区靠传感器，文本不可推断，不计分
                    continue
                if k == "query":
                    if not _semantic_query_ok(want, have):
                        sem_ok = False
                elif k == "value" and got.get("function") == "temperature":
                    if _TEMP_EQ.get(str(want).lower(), {str(want).lower()}) != \
                       _TEMP_EQ.get(have.lower(), {have.lower()}):
                        sem_ok = False
                elif have.lower() != str(want).lower():
                    sem_ok = False
            param_hit, param_hit_exact = sem_ok, exact_ok
    actual = " | ".join("%s(%s)" % (c["name"], ",".join("%s=%s" % kv for kv in c["arguments"].items()))
                        for c in calls) or (((content or "")[:80]).replace("\n", " ") or "<no call>")
    return {"name_hit": name_hit, "param_hit": param_hit, "param_hit_exact": param_hit_exact,
            "actual": actual, "no_call": not calls}


# ============================ 评分段：VQA/闲聊 gpt-oss-120b judge ============================
# 内联自上游基准 score.py:P_REF / P_NOREF / judge_one（prompt 逐字节一致 + robust 解析）。

P_REF = """你是评测判分员。下面是一个车载语音助手的问答，以及一条或多条参考答案。
参考答案可能带不同的说话风格（幽默/详尽/简洁等），**风格差异不算错**。
请只判断「模型回答在事实上是否与参考答案一致」。

判定规则：
- correct：模型回答的关键事实与任一参考答案一致（数量、物体、颜色、方位、有无等核心信息对得上）。允许更简洁或更详细、允许措辞不同。
- partial：部分关键事实对、但有一处明显不符或遗漏了问题所问的核心信息。
- wrong：关键事实与参考答案矛盾，或答非所问，或编造了参考答案中没有且与之冲突的内容。
- refuse：模型没有回答内容本身（表示看不到/无法处理/反问而不作答）。

问题：{q}
参考答案：
{refs}
模型回答：{pred}

只输出一行 JSON：{{"verdict":"correct|partial|wrong|refuse","reason":"不超过30字"}}"""

P_NOREF = """你是评测判分员。下面是一个车载语音助手的问答，没有参考答案。
请判断模型回复是否是一个合理可用的回复。

判定规则：
- correct：回复切题、内容无明显事实错误、符合车载助手口语化简洁的要求。
- partial：切题但过于空泛/啰嗦/有小瑕疵。
- wrong：答非所问，或有明显事实错误。
- refuse：拒答或声称无法处理（对普通问答而言不合理）。

问题：{q}
模型回答：{pred}

只输出一行 JSON：{{"verdict":"correct|partial|wrong|refuse","reason":"不超过30字"}}"""

_VALID = ("correct", "partial", "wrong", "refuse")


def _question_only(text: str) -> str:
    """从 VQA persona 全文里抠出问题行；闲聊直接是纯文本。"""
    m = re.search(r'问题："(.*?)"', text or "", re.S)
    return (m.group(1) if m else (text or ""))[:500]


def _user_text(messages: list[dict]) -> str:
    c = messages[-1].get("content")
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")


def judge_one(q: str, refs: list[str], pred: str, timeout: int = 120) -> dict:
    import urllib.request
    if not (pred or "").strip():
        return {"verdict": "refuse", "reason": "空回复"}
    prompt = (P_REF.format(q=q, refs="\n".join("- " + r for r in refs), pred=pred)
              if refs else P_NOREF.format(q=q, pred=pred))
    body = json.dumps({"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": 2000}).encode()
    req = urllib.request.Request(JUDGE_URL, data=body, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        msg = json.loads(resp.read())["choices"][0]["message"]
    txt = (msg.get("content") or "").strip()
    if not txt:  # 预算被 reasoning 吃光时答案可能只在 reasoning 里
        txt = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    m = re.search(r"\{.*?\}", txt, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).lower()
            if v in _VALID:
                return {"verdict": v, "reason": str(d.get("reason", ""))[:60]}
        except Exception:
            pass
    m2 = re.search(r'"verdict"\s*:\s*"(\w+)"', txt)      # JSON 截断兜底
    if m2 and m2.group(1).lower() in _VALID:
        return {"verdict": m2.group(1).lower(), "reason": "(截断兜底)"}
    return {"verdict": "parse_error", "reason": txt[:60]}


# ============================ 评分驱动 + 汇总 ============================

# 发布版数据集的四个标准任务名（category 无前缀）。
_TASKS = ("vqa_incar", "vqa_outcar", "chat", "toolcall")


def _cat_key(cat: str) -> str:
    """归一化 category → 标准任务名。

    发布版 category 本就是 _TASKS 之一；对可能带前缀的历史数据（形如 <prefix>_toolcall）
    按后缀匹配剥回标准任务名，保证判分/汇总口径一致。
    """
    cat = cat or ""
    for t in _TASKS:
        if cat == t or cat.endswith("_" + t):
            return t
    return cat


def score(samples: list[dict], preds: list[dict], out_path: Path, concurrency: int = 8) -> dict:
    ev = {s["sample_id"]: s for s in samples}
    rows = []
    judge_jobs = []               # (pred, sample) 待 judge 的 VQA/闲聊
    for p in preds:
        sid = p.get("sample_id")
        rec = ev.get(sid)
        if not rec:
            continue
        cat = rec["category"]
        if _cat_key(cat) == "toolcall":
            r = judge_toolcall(rec["expect"], p.get("content") or "", p.get("tool_calls"),
                               rec["messages"][-1].get("content"))
            rows.append({"sample_id": sid, "category": cat, "channel": "toolcall", **r,
                         "exclude": rec["meta"].get("exclude_name_scoring", False)})
        else:
            judge_jobs.append((p, rec))

    judge_on = bool(KEY)
    if judge_jobs and not judge_on:
        print("⚠️  未提供 MODELS_PROXY_API_KEY：VQA/闲聊 judge 跳过（toolcall 已正常打分）", flush=True)
        for p, rec in judge_jobs:
            rows.append({"sample_id": p["sample_id"], "category": rec["category"],
                         "channel": "judge", "verdict": "skip", "reason": "no_key",
                         "has_ref": bool(rec.get("gt_candidates"))})
    elif judge_jobs:
        import concurrent.futures as cf, threading
        lock = threading.Lock()
        done = [0]

        def work(item):
            p, rec = item
            refs = [g for g in (rec.get("gt_candidates") or []) if (g or "").strip()]
            q = rec["meta"].get("query") or _question_only(_user_text(rec["messages"]))
            for att in range(3):
                try:
                    v = judge_one(q, refs, p.get("content") or "")
                    break
                except Exception as e:
                    if att == 2:
                        v = {"verdict": "judge_error", "reason": str(e)[:60]}
                    else:
                        time.sleep(2 + 3 * att)
            with lock:
                done[0] += 1
                if done[0] % 20 == 0 or done[0] == len(judge_jobs):
                    print(f"  judged {done[0]}/{len(judge_jobs)}", flush=True)
            return {"sample_id": p["sample_id"], "category": rec["category"], "channel": "judge",
                    "verdict": v["verdict"], "reason": v.get("reason", ""), "has_ref": bool(refs)}

        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            rows.extend(ex.map(work, judge_jobs))

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    summ = {}
    for cat, rs in sorted(by_cat.items()):
        if _cat_key(cat) == "toolcall":
            scored = [r for r in rs if not r.get("exclude")]
            pc = [r for r in scored if r.get("param_hit") is not None]
            pe = [r for r in scored if r.get("param_hit_exact") is not None]
            summ[cat] = {"n": len(rs), "scored": len(scored),
                         "name_hit": sum(1 for r in scored if r.get("name_hit")),
                         "no_call": sum(1 for r in rs if r.get("no_call")),
                         "param_semantic": sum(1 for r in pc if r["param_hit"]),
                         "param_semantic_n": len(pc),
                         "param_exact": sum(1 for r in pe if r["param_hit_exact"]),
                         "param_exact_n": len(pe),
                         "name_hit_pct": round(100.0 * sum(1 for r in scored if r.get("name_hit"))
                                               / max(1, len(scored)), 2)}
        else:
            c = collections.Counter(r["verdict"] for r in rs)
            n = len(rs)
            scored_n = n - c["skip"]                     # skip 不计入分母
            summ[cat] = {"n": n, "correct": c["correct"], "partial": c["partial"],
                         "wrong": c["wrong"], "refuse": c["refuse"], "skip": c["skip"],
                         "err": c["parse_error"] + c["judge_error"],
                         "acc": round(100.0 * c["correct"] / scored_n, 2) if scored_n else None,
                         "acc_loose": round(100.0 * (c["correct"] + c["partial"]) / scored_n, 2)
                         if scored_n else None}
    return summ


# ============================ --compare 汇总表 ============================

def compare() -> int:
    tags = sorted(d for d in RESULTS_DIR.glob("*") if (d / "summary.json").is_file())
    if not tags:
        print(f"没有可对比的结果（{RESULTS_DIR}/*/summary.json 为空）", file=sys.stderr)
        return 1
    data = {d.name: {_cat_key(k): v for k, v in
                     json.loads((d / "summary.json").read_text("utf-8")).get("by_category", {}).items()}
            for d in tags}
    names = list(data)
    CATS = [("vqa_incar", "VQA·车内", "vqa"), ("vqa_outcar", "VQA·车外", "vqa"),
            ("chat", "闲聊", "vqa"), ("toolcall", "ToolCall", "tool")]

    def cell(summ, cat, kind):
        s = summ.get(cat)
        if not s:
            return "-"
        if kind == "tool":
            return f"名{s['name_hit']}/{s['scored']} 参{s['param_semantic']}/{s['param_semantic_n']}"
        if s.get("acc") is None:
            return "skip(无key)"
        return f"{s['acc']}% / {s['acc_loose']}%"

    print("\n## Query_06_cabin 模型分数对比\n")
    print(f"VQA/闲聊：correct% / 宽松%（judge={JUDGE_MODEL}）；ToolCall：工具名命中 / 参数语义命中\n")
    print("| 子任务 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for cat, label, kind in CATS:
        print(f"| {label} | " + " | ".join(cell(data[n], cat, kind) for n in names) + " |")
    print()
    return 0


# ============================ main ============================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="本地 Qwen3-VL HF 目录")
    ap.add_argument("--tag", help="结果子目录名（默认取 --model 末段）")
    ap.add_argument("--eval", default=str(DATA_DIR / "eval.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 条（冒烟）")
    ap.add_argument("--concurrency", type=int, default=8, help="judge 并发数")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--enable-thinking", action="store_true", help="开启 Thinking 解码（默认关）")
    ap.add_argument("--score-only", action="store_true", help="跳过推理，只对已有 outputs 重打分")
    ap.add_argument("--compare", action="store_true", help="汇总 results/*/summary.json 成对比表")
    a = ap.parse_args()

    if a.compare:
        return compare()
    if not a.model:
        ap.error("需要 --model（或用 --compare）")

    eval_path = Path(a.eval)
    if not eval_path.is_file():
        print(f"缺 {eval_path}，请先跑 build_dataset.py", file=sys.stderr)
        return 1
    samples = [json.loads(l) for l in eval_path.open(encoding="utf-8")]
    if a.limit:
        samples = samples[:a.limit]

    tag = a.tag or Path(a.model.rstrip("/")).name
    run_dir = RESULTS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path, score_path = run_dir / "outputs.jsonl", run_dir / "scores.jsonl"

    print(f"=== Query_06_cabin | tag={tag} | {len(samples)} 条 ===", flush=True)
    if not a.score_only:
        run_inference(a.model, samples, out_path, dtype=a.dtype, enable_thinking=a.enable_thinking)
    elif not out_path.is_file():
        print(f"--score-only 但缺 {out_path}", file=sys.stderr)
        return 1

    preds = [json.loads(l) for l in out_path.open(encoding="utf-8")]
    print(f"[score] {len(preds)} 条 …（toolcall 程序判分；VQA/闲聊 judge={'on' if KEY else 'off(无key)'}）", flush=True)
    summ = score(samples, preds, score_path, concurrency=a.concurrency)
    summ_full = {"tag": tag, "model": a.model, "n": len(samples),
                 "judge_model": JUDGE_MODEL if KEY else None, "by_category": summ}
    (run_dir / "summary.json").write_text(json.dumps(summ_full, ensure_ascii=False, indent=2), "utf-8")
    print("\n" + json.dumps(summ, ensure_ascii=False, indent=2))
    print(f"\n结果 -> {run_dir}/（outputs / scores / summary）")
    print("对比两模型： python3 run_eval.py --compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
