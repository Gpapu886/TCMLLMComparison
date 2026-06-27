print("STEP 0: script started", flush=True)

import json
import argparse
from pathlib import Path
print("STEP 1: basic imports ok", flush=True)

import torch
print("STEP 2: torch import ok", flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
print("STEP 3: transformers import ok", flush=True)

from peft import PeftModel
print("STEP 4: peft import ok", flush=True)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

SYSTEM_PROMPT = "你是一名严谨的医学人工智能助手。请根据用户问题作答。答案应直接、完整、清晰。"

def apply_chat_template_safe(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
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

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def count_finished(path):
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                n += 1
            except Exception:
                break
    return n

@torch.inference_mode()
def generate_answer(model, tokenizer, question, max_input_tokens, max_new_tokens):
    prompt = apply_chat_template_safe(tokenizer, question)

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
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return answer.strip()

def main():
    print("STEP 5: entering main", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit_files", type=int, default=-1)
    parser.add_argument("--limit_samples_per_file", type=int, default=-1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.jsonl"))
    if args.limit_files > 0:
        files = files[:args.limit_files]

    print("Input dir:", input_dir, flush=True)
    print("Output dir:", output_dir, flush=True)
    print("Files:", len(files), flush=True)
    print("Base model:", args.base_model, flush=True)
    print("Adapter:", args.adapter_path, flush=True)

    print("STEP 6: loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("STEP 7: tokenizer loaded", flush=True)

    print("STEP 8: loading base model", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).cuda()
    print("STEP 9: base model loaded", flush=True)

    print("STEP 10: loading LoRA adapter", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.eval()
    print("STEP 11: LoRA model ready", flush=True)

    with open(output_dir / "infer_manifest.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    for file_path in files:
        out_path = output_dir / file_path.name
        items = load_jsonl(file_path)

        if args.limit_samples_per_file > 0:
            items = items[:args.limit_samples_per_file]

        done = count_finished(out_path)
        print(f"Processing {file_path.name}: total={len(items)}, done={done}", flush=True)

        with open(out_path, "a", encoding="utf-8") as fout:
            for i in tqdm(range(done, len(items)), desc=file_path.name):
                obj = items[i]
                question = obj.get("question", "")
                if not isinstance(question, str):
                    question = str(question)

                answer = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=args.max_new_tokens,
                )

                obj["answer"] = answer
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fout.flush()

        print("Finished file:", file_path.name, flush=True)

    print("ALL_INFERENCE_FINISHED", flush=True)

if __name__ == "__main__":
    main()
