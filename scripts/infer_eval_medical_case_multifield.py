import json
import re
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


SECTION_NAMES = ["证候", "病因", "诊断", "治宜", "推荐"]


def norm_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", "", s)
    s = s.replace("：", ":")
    s = s.strip()
    return s


def extract_sections(text):
    text = str(text or "")
    sections = {k: "" for k in SECTION_NAMES}

    # 优先匹配【证候】这种格式
    for i, name in enumerate(SECTION_NAMES):
        start_pat = rf"【{re.escape(name)}】"
        m = re.search(start_pat, text)
        if not m:
            # 兼容 证候： 或 证候:
            m = re.search(rf"{re.escape(name)}[：:]", text)
        if not m:
            continue

        start = m.end()
        end = len(text)

        for next_name in SECTION_NAMES:
            if next_name == name:
                continue
            m2 = re.search(rf"【{re.escape(next_name)}】|{re.escape(next_name)}[：:]", text[start:])
            if m2:
                end = min(end, start + m2.start())

        sections[name] = text[start:end].strip()

    return sections


def split_recommendation_items(s):
    s = str(s or "")
    s = re.sub(r"常见中药有|方用|推荐|包括|可用|可选|中药有", "", s)
    parts = re.split(r"[、，,；;。\n\s]+", s)
    items = []
    for x in parts:
        x = x.strip()
        if len(x) >= 2:
            items.append(x)
    return set(items)


def score_one(gold_text, pred_text):
    gold = extract_sections(gold_text)
    pred = extract_sections(pred_text)

    metrics = {}

    for gname, key_prefix in [
        ("证候", "syndrome"),
        ("诊断", "diagnosis"),
        ("治宜", "treatment"),
    ]:
        g = norm_text(gold.get(gname, ""))
        p = norm_text(pred.get(gname, ""))

        exact = int(g != "" and p == g)
        contains = int(g != "" and (g in p or p in g) and p != "")

        metrics[f"{key_prefix}_exact"] = exact
        metrics[f"{key_prefix}_contains"] = contains

    gold_items = split_recommendation_items(gold.get("推荐", ""))
    pred_items = split_recommendation_items(pred.get("推荐", ""))

    if gold_items and pred_items:
        inter = gold_items & pred_items
        precision = len(inter) / len(pred_items) if pred_items else 0.0
        recall = len(inter) / len(gold_items) if gold_items else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        hit = int(len(inter) > 0)
    else:
        precision = recall = f1 = 0.0
        hit = 0

    metrics["recommendation_hit"] = hit
    metrics["recommendation_precision"] = precision
    metrics["recommendation_recall"] = recall
    metrics["recommendation_f1"] = f1

    metrics["full_output_exact"] = int(
        metrics["syndrome_exact"] == 1 and
        metrics["diagnosis_exact"] == 1 and
        metrics["treatment_exact"] == 1 and
        metrics["recommendation_hit"] == 1
    )

    return gold, pred, metrics


def apply_chat_template_safe(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.inference_mode()
def generate_answer(model, tokenizer, messages, max_input_tokens, max_new_tokens):
    prompt = apply_chat_template_safe(tokenizer, messages)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def load_done_ids(raw_eval_path):
    if not raw_eval_path.exists():
        return set()
    done = set()
    try:
        obj = json.load(open(raw_eval_path, "r", encoding="utf-8"))
        for r in obj.get("results", []):
            done.add(r["id"])
    except Exception:
        pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--raw_eval_file", required=True)
    parser.add_argument("--multifield_file", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    args = parser.parse_args()

    test_file = Path(args.test_file)
    raw_eval_file = Path(args.raw_eval_file)
    multifield_file = Path(args.multifield_file)
    raw_eval_file.parent.mkdir(parents=True, exist_ok=True)
    multifield_file.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit > 0:
        samples = samples[:args.limit]

    print("TEST_FILE:", test_file, flush=True)
    print("N:", len(samples), flush=True)
    print("RAW_EVAL_FILE:", raw_eval_file, flush=True)
    print("MULTIFIELD_FILE:", multifield_file, flush=True)
    print("BASE_MODEL:", args.base_model, flush=True)
    print("ADAPTER:", args.adapter_path if args.adapter_path else "NONE_BASE_ONLY", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).cuda()

    if args.adapter_path:
        print("Loading LoRA adapter...", flush=True)
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    model.eval()
    print("Model ready.", flush=True)

    # 断点续跑
    raw_results = []
    if raw_eval_file.exists():
        try:
            old = json.load(open(raw_eval_file, "r", encoding="utf-8"))
            raw_results = old.get("results", [])
            print("Resume old results:", len(raw_results), flush=True)
        except Exception:
            raw_results = []

    done_ids = set(r["id"] for r in raw_results if "id" in r)

    for sample in tqdm(samples, desc="infer"):
        sid = sample.get("id", "")
        if sid in done_ids:
            continue

        messages = sample["messages"]
        gold = sample.get("raw_output", "")

        try:
            pred = generate_answer(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            pred = "[GENERATION_ERROR] " + repr(e)

        raw_results.append({
            "id": sid,
            "gold": gold,
            "prediction": pred,
            "syndrome": sample.get("syndrome", ""),
            "raw_input": sample.get("raw_input", ""),
        })

        # 每条保存一次，防止中断损失
        with open(raw_eval_file, "w", encoding="utf-8") as f:
            json.dump({
                "test_file": str(test_file),
                "base_model": args.base_model,
                "adapter_path": args.adapter_path,
                "n": len(raw_results),
                "results": raw_results,
            }, f, ensure_ascii=False, indent=2)

    # 多字段评分
    scored = []
    sums = {
        "syndrome_exact": 0,
        "syndrome_contains": 0,
        "diagnosis_exact": 0,
        "diagnosis_contains": 0,
        "treatment_exact": 0,
        "treatment_contains": 0,
        "recommendation_hit": 0,
        "recommendation_precision": 0.0,
        "recommendation_recall": 0.0,
        "recommendation_f1": 0.0,
        "full_output_exact": 0,
    }

    for r in raw_results:
        gold_sections, pred_sections, metrics = score_one(r["gold"], r["prediction"])
        for k in sums:
            sums[k] += metrics[k]
        scored.append({
            "id": r["id"],
            "gold_sections": gold_sections,
            "pred_sections": pred_sections,
            "metrics": metrics,
        })

    n = len(raw_results)
    summary = {
        "n": n,
        "syndrome_exact_accuracy": sums["syndrome_exact"] / n if n else 0,
        "syndrome_contains_accuracy": sums["syndrome_contains"] / n if n else 0,
        "diagnosis_exact_accuracy": sums["diagnosis_exact"] / n if n else 0,
        "diagnosis_contains_accuracy": sums["diagnosis_contains"] / n if n else 0,
        "treatment_exact_accuracy": sums["treatment_exact"] / n if n else 0,
        "treatment_contains_accuracy": sums["treatment_contains"] / n if n else 0,
        "recommendation_hit_rate": sums["recommendation_hit"] / n if n else 0,
        "recommendation_precision": sums["recommendation_precision"] / n if n else 0,
        "recommendation_recall": sums["recommendation_recall"] / n if n else 0,
        "recommendation_f1": sums["recommendation_f1"] / n if n else 0,
        "full_output_exact_accuracy": sums["full_output_exact"] / n if n else 0,
    }

    with open(multifield_file, "w", encoding="utf-8") as f:
        json.dump({
            "source_eval_file": str(raw_eval_file),
            "summary": summary,
            "results": scored,
        }, f, ensure_ascii=False, indent=2)

    print("SUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
