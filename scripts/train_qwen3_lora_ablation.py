import os
import json
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = (
    "You are a traditional Chinese medicine assistant. "
    "Answer the user's question according to the given instruction and input. "
    "The response is for research evaluation only."
)


def apply_chat_template_safe(tokenizer, messages, add_generation_prompt=False):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def load_records(data_dir, max_samples_per_file=-1):
    data_dir = Path(data_dir)
    files = sorted([p for p in data_dir.glob("*.json")])
    if not files:
        raise FileNotFoundError(f"No json files found in {data_dir}")

    all_records = []
    file_counts = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Skip non-list json: {path}")
            continue

        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            instruction = str(item.get("instruction", "")).strip()
            inp = str(item.get("input", "")).strip()
            out = str(item.get("output", "")).strip()

            if not out:
                continue

            records.append({
                "source_file": path.name,
                "instruction": instruction,
                "input": inp,
                "output": out,
            })

        if max_samples_per_file and max_samples_per_file > 0:
            random.shuffle(records)
            records = records[:max_samples_per_file]

        file_counts[path.name] = len(records)
        all_records.extend(records)

    return all_records, file_counts


class SFTDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos = tokenizer.eos_token or ""

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        user_content = ""
        if r["instruction"]:
            user_content += r["instruction"].strip()
        if r["input"]:
            if user_content:
                user_content += "\n\n"
            user_content += r["input"].strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt_text = apply_chat_template_safe(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
        )

        answer_text = r["output"].strip()
        if self.eos and not answer_text.endswith(self.eos):
            answer_text += self.eos

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"]

        # Keep answer tokens; truncate prompt from the left if needed.
        if len(answer_ids) >= self.max_length:
            answer_ids = answer_ids[: self.max_length - 1]

        max_prompt_len = self.max_length - len(answer_ids)
        if max_prompt_len <= 0:
            prompt_ids = []
        elif len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        for x in features:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_samples_per_file", type=int, default=-1)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=500)

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=20)

    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading records...")
    records, file_counts = load_records(args.data_dir, args.max_samples_per_file)
    random.shuffle(records)

    if args.max_train_samples and args.max_train_samples > 0:
        records = records[: args.max_train_samples + args.max_eval_samples]

    eval_n = min(args.max_eval_samples, max(1, len(records) // 100))
    eval_records = records[:eval_n]
    train_records = records[eval_n:]

    print("File counts:", file_counts)
    print("Train records:", len(train_records))
    print("Eval records:", len(eval_records))

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_dir": args.data_dir,
            "seed": args.seed,
            "file_counts": file_counts,
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "max_length": args.max_length,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_samples_per_file": args.max_samples_per_file,
            "max_train_samples": args.max_train_samples,
        }, f, ensure_ascii=False, indent=2)

    train_dataset = SFTDataset(train_records, tokenizer, args.max_length)
    eval_dataset = SFTDataset(eval_records, tokenizer, args.max_length)
    collator = DataCollator(tokenizer)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=args.seed,
        data_seed=args.seed,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        bf16=True,
        fp16=False,
        gradient_checkpointing=True,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=2,

        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Training finished. Saved to:", output_dir)


if __name__ == "__main__":
    main()
