import os
import re
import json
import math
import random
import argparse
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def normalize_answer(s: str) -> str:
    s = (s or "").strip()
    s = s.replace(",", "")
    return s


def extract_gold(answer_text: str) -> str:
    m = re.search(r"####\s*([^\n\r]+)", answer_text or "")
    if m:
        return normalize_answer(m.group(1))
    return normalize_answer((answer_text or "").strip().split("\n")[-1])


def extract_pred(text: str) -> str:
    t = (text or "").strip()

    # 1) Prefer the last "#### <answer>" if present.
    matches = re.findall(r"####\s*([^\n\r]+)", t)
    if matches:
        return normalize_answer(matches[-1])

    # 2) Check the last few non-empty lines for a numeric tail.
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    for ln in reversed(lines[-5:]):
        m = re.search(r"(-?\d[\d,]*\.?\d*)\s*$", ln)
        if m:
            return normalize_answer(m.group(1))

    # 3) Fallback: last numeric token in the whole text.
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    if nums:
        return normalize_answer(nums[-1])

    # 4) Last-resort fallback.
    return normalize_answer(lines[-1] if lines else "")


def build_fewshot_block(examples: List[Dict[str, str]]) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        q = ex["question"].strip()
        a = ex["answer"].strip()
        lines.append(f"Example {i}:\nQuestion: {q}\nAnswer: {a}\n")
    return "\n".join(lines).strip()


def build_messages_user(question: str, fewshot_block: str, system_prompt: str) -> List[Dict[str, str]]:
    user_text = (
        f"Here are some solved examples:\n\n{fewshot_block}\n\n"
        f"Now solve the following problem.\nQuestion: {question.strip()}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def unwrap_model(m):
    if hasattr(m, "module"):
        return m.module
    if hasattr(m, "unwrap_model") and callable(getattr(m, "unwrap_model")):
        try:
            return m.unwrap_model()
        except Exception:
            pass
    return m


def load_fixed_fewshot(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "examples" not in obj:
        raise ValueError(f"Invalid fewshot json format in {path}")
    exs = obj["examples"]
    if not isinstance(exs, list) or len(exs) == 0:
        raise ValueError(f"'examples' is empty in {path}")
    for i, x in enumerate(exs):
        if "question" not in x or "answer" not in x:
            raise ValueError(f"fewshot example[{i}] missing question/answer in {path}")
    return exs


def sample_fewshot_from_official_train(train_rows, n_shot: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    indices = list(range(len(train_rows)))
    rng.shuffle(indices)
    pick = indices[: min(n_shot, len(indices))]
    return [{"question": train_rows[i]["question"], "answer": train_rows[i]["answer"]} for i in pick]


def choose_dtype(args) -> torch.dtype:
    """
    Prefer bf16 when available.
    Priority:
    1) explicit --bf16 / --fp16
    2) auto bf16 if cuda bf16 supported
    3) else fp16 if cuda
    4) else fp32 on cpu
    """
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")

    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            print("[INFO] Auto dtype: bfloat16 (CUDA bf16 supported)")
            return torch.bfloat16
        print("[INFO] Auto dtype: float16 (CUDA bf16 not supported)")
        return torch.float16

    print("[INFO] Auto dtype: float32 (CPU)")
    return torch.float32


def cast_model_dtype_inplace(model: torch.nn.Module, target_dtype: torch.dtype) -> None:
    """
    Force all floating-point parameters/buffers to target dtype.
    This prevents runtime dtype mismatch like BF16 vs FP32 after PEFT load/merge.
    """
    for p in model.parameters():
        if p.is_floating_point() and p.dtype != target_dtype:
            p.data = p.data.to(dtype=target_dtype)
    for name, b in model.named_buffers():
        if b is not None and b.is_floating_point() and b.dtype != target_dtype:
            setattr(model, name, b.to(dtype=target_dtype))


@torch.no_grad()
def evaluate_gsm8k_test(
    model,
    tokenizer,
    test_rows: List[Dict[str, str]],
    fewshot_block: str,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int,
    save_path: str,
) -> Tuple[float, int, int]:
    gen_model = unwrap_model(model)

    old_padding_side = tokenizer.padding_side
    was_training = gen_model.training

    old_cfg_use_cache = getattr(gen_model.config, "use_cache", None)
    old_gen_use_cache = None
    if hasattr(gen_model, "generation_config") and gen_model.generation_config is not None:
        old_gen_use_cache = getattr(gen_model.generation_config, "use_cache", None)

    correct = 0
    total = 0
    num_batches = math.ceil(len(test_rows) / batch_size)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fout = open(save_path, "w", encoding="utf-8")

    try:
        tokenizer.padding_side = "left"
        gen_model.eval()

        if hasattr(gen_model.config, "use_cache"):
            gen_model.config.use_cache = True
        if hasattr(gen_model, "generation_config") and gen_model.generation_config is not None:
            gen_model.generation_config.use_cache = True

        pbar = tqdm(
            range(0, len(test_rows), batch_size),
            total=num_batches,
            desc="GSM8K test",
            leave=True,
        )

        for start in pbar:
            batch_rows = test_rows[start: start + batch_size]
            questions = [r["question"] for r in batch_rows]
            golds = [extract_gold(r["answer"]) for r in batch_rows]

            prompts = []
            for q in questions:
                messages = build_messages_user(q, fewshot_block, system_prompt=system_prompt)
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(gen_model.device)

            prompt_len = inputs["input_ids"].shape[1]

            out = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

            for i, row in enumerate(batch_rows):
                gen_ids = out[i][prompt_len:]
                pred_full = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred = extract_pred(pred_full)
                gold = golds[i]
                is_correct = int(pred == gold)

                correct += is_correct
                total += 1

                rec = {
                    "idx": start + i,
                    "question": row["question"],
                    "gold_full": row["answer"],
                    "gold": gold,
                    "pred_full": pred_full,
                    "pred": pred,
                    "correct": is_correct,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            pbar.set_postfix({"acc": f"{(correct / total) if total else 0.0:.4f}"})

    finally:
        fout.close()
        tokenizer.padding_side = old_padding_side

        if old_cfg_use_cache is not None and hasattr(gen_model.config, "use_cache"):
            gen_model.config.use_cache = old_cfg_use_cache
        if (
            old_gen_use_cache is not None
            and hasattr(gen_model, "generation_config")
            and gen_model.generation_config is not None
        ):
            gen_model.generation_config.use_cache = old_gen_use_cache

        if was_training:
            gen_model.train()

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="", help="Path to LoRA adapter dir (optional)")
    parser.add_argument("--merge_adapter", action="store_true", help="Merge adapter into base model for faster inference")
    parser.add_argument("--output_dir", type=str, default="out_eval_medium")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful math assistant. Solve math problems by showing your reasoning steps and end with the final answer in the format: #### <answer>",
    )

    parser.add_argument("--fewshot_file", type=str, default="", help="Path to fixed_fewshot_examples.json (strongly recommended)")
    parser.add_argument("--n_shot", type=int, default=6, help="Used only if fewshot_file is not provided")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=3072)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="",
        choices=["", "flash_attention_2", "sdpa", "eager"],
        help="Optional attention implementation for from_pretrained",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load GSM8K
    ds = load_dataset("gsm8k", "main")
    official_train = ds["train"]
    official_test = ds["test"]

    # 2) Few-shot setup
    if args.fewshot_file:
        fewshot_examples = load_fixed_fewshot(args.fewshot_file)
        fewshot_source = f"fixed file: {args.fewshot_file}"
    else:
        fewshot_examples = sample_fewshot_from_official_train(official_train, args.n_shot, args.seed)
        fewshot_source = f"sampled from official train (n_shot={len(fewshot_examples)}, seed={args.seed})"
        print(
            "[WARN] --fewshot_file not provided. Using sampled few-shot examples.\n"
            "[WARN] This may reduce reproducibility across runs/environments.\n"
            "[WARN] For stable comparison, pass training artifact fixed_fewshot_examples.json."
        )

    fewshot_block = build_fewshot_block(fewshot_examples)

    used_fewshot_path = os.path.join(args.output_dir, "test_used_fewshot_examples.json")
    with open(used_fewshot_path, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": args.seed, "source": fewshot_source, "examples": fewshot_examples},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 3) Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = choose_dtype(args)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except Exception as e:
        if args.attn_implementation:
            print(f"[WARN] Failed with attn_implementation={args.attn_implementation}: {repr(e)}")
            print("[WARN] Retry without explicit attn_implementation.")
            model_kwargs.pop("attn_implementation", None)
            base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        else:
            raise

    # Force base model dtype (safety)
    cast_model_dtype_inplace(base_model, torch_dtype)

    if args.adapter_dir:
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        # Safety: adapter load may introduce fp32 weights
        cast_model_dtype_inplace(model, torch_dtype)

        model_source = f"base + adapter: {args.adapter_dir}"

        if args.merge_adapter:
            try:
                model = model.merge_and_unload()
                # Safety: merge may produce fp32 tensors
                cast_model_dtype_inplace(model, torch_dtype)
                model_source = f"base + adapter(merged): {args.adapter_dir}"
                print("[INFO] Adapter merged into base model via merge_and_unload().")
            except Exception as e:
                print(f"[WARN] merge_and_unload failed, keep PEFT model. reason={repr(e)}")
    else:
        model = base_model
        model_source = "base model only"

    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    if hasattr(model, "generation_config") and model.generation_config is not None:
        gc = model.generation_config
        gc.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            gc.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            gc.bos_token_id = tokenizer.bos_token_id
        gc.do_sample = False
        if hasattr(gc, "temperature"):
            gc.temperature = None
        if hasattr(gc, "top_p"):
            gc.top_p = None
        if hasattr(gc, "top_k"):
            gc.top_k = None

    # 4) Evaluate
    test_rows = [official_test[i] for i in range(len(official_test))]
    pred_path = os.path.join(args.output_dir, "pred_gsm8k.jsonl")

    acc, correct, total = evaluate_gsm8k_test(
        model=model,
        tokenizer=tokenizer,
        test_rows=test_rows,
        fewshot_block=fewshot_block,
        system_prompt=args.system_prompt,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        save_path=pred_path,
    )

    # 5) Save summary
    summary = {
        "model_name": args.model_name,
        "model_source": model_source,
        "adapter_dir": args.adapter_dir,
        "merge_adapter": args.merge_adapter,
        "fewshot_source": fewshot_source,
        "fewshot_file": args.fewshot_file,
        "used_fewshot_saved_to": used_fewshot_path,
        "system_prompt": args.system_prompt,
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "dtype": str(torch_dtype),
        "attn_implementation": args.attn_implementation if args.attn_implementation else "default",
        "correct": correct,
        "total": total,
        "test_acc": acc,
        "pred_file": pred_path,
    }

    summary_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Model source: {model_source}")
    print(f"[INFO] Few-shot source: {fewshot_source}")
    print(f"[DONE] Saved predictions: {pred_path}")
    print(f"[DONE] Saved metrics: {summary_path}")
    print(f"[TEST ACC] {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
