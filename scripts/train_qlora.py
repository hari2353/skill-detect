"""Real QLoRA SFT with TRL (API verified against TRL 1.x SFT docs).

Proof default (ungated, tiny): HuggingFaceTB/SmolLM2-135M-Instruct — proves the
full GPU loop on 8GB in minutes.
Target per configs/lora.yaml: google/gemma-3-4b-it (gated: `hf auth login` +
accept the license; 4-bit needs bitsandbytes, which is Linux-first — on native
Windows use --quant none for ≤1B proof models or WSL2 for the 4B QLoRA run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill_project.training import LoRAConfig


def load_chat_dataset(path: str):
    from datasets import Dataset
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))  # {"messages": [...]} conversational LM
    if not rows:
        raise ValueError(f"no records in {path} — run build_dataset.py first")
    return Dataset.from_list(rows)


def build_quant_config():
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "4-bit needs bitsandbytes (pip install bitsandbytes). "
            "Native Windows is unsupported — use WSL2, or --quant none."
        ) from exc
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes not importable on this platform. "
            "Use WSL2 for 4-bit, or --quant none for ≤1B proof models."
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora.yaml")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--output", default="adapters/smollm2-proof")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--quant", choices=["4bit", "none"], default="none")
    ap.add_argument("--assistant-only", dest="assistant_only", action="store_true", default=True)
    ap.add_argument("--no-assistant-only", dest="assistant_only", action="store_false",
                    help="if the model's chat template lacks generation markers")
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, TaskType
    from trl import SFTConfig, SFTTrainer

    cfg = LoRAConfig.from_yaml(args.config)
    ds = load_chat_dataset(args.train)
    print(f"records={len(ds)} model={args.model} quant={args.quant}")

    model_kwargs = {"dtype": torch.float16}
    bnb = None
    if args.quant == "4bit":
        bnb = build_quant_config()
        model_kwargs["device_map"] = "auto"

    sft = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=cfg.grad_checkpointing,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_steps=args.max_steps,
        save_total_limit=1,
        fp16=True,
        report_to="none",
        seed=7,
        max_length=cfg.max_seq_len,
        assistant_only_loss=args.assistant_only,
        model_init_kwargs=model_kwargs,
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )
    trainer = SFTTrainer(model=args.model, args=sft, train_dataset=ds,
                         peft_config=lora, quantization_config=bnb)
    out = trainer.train()
    trainer.save_model(args.output)
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    print(json.dumps({"global_steps": out.global_step,
                      "train_loss": round(out.training_loss, 4),
                      "first_loss": round(losses[0], 4) if losses else None,
                      "last_loss": round(losses[-1], 4) if losses else None,
                      "adapter_dir": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
