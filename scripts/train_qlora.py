"""LoRA SFT with plain transformers Trainer + PEFT (no TRL fused-kernel path).

Why not TRL SFTTrainer here: TRL 1.x unconditionally patches a triton fused LM
head; triton has no Windows wheels, so the fused path crashes with
`'NoneType' has no .apply` on native Windows. Same math (token NLL, full
sequence), zero fused kernels. Revisit TRL inside WSL2.

Proof default (ungated, tiny): HuggingFaceTB/SmolLM2-135M-Instruct — full GPU
loop on 8GB in minutes.
Target per configs/lora.yaml: google/gemma-3-4b-it (gated: `hf auth login` +
accept the license; 4-bit needs bitsandbytes, Linux-first — native Windows
uses --quant none for small proofs, WSL2 for the 4B QLoRA run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill_project.training import LoRAConfig


def load_text_dataset(path: str, tokenizer, max_len: int):
    from datasets import Dataset
    texts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                msgs = json.loads(line)["messages"]
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
    if not texts:
        raise ValueError(f"no records in {path} — run build_dataset.py first")
    ds = Dataset.from_dict({"text": texts})

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    return ds.map(tok, batched=True, remove_columns=["text"])


def build_quant_config():
    try:
        import torch
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "4-bit needs bitsandbytes (pip install bitsandbytes). "
            "Native Windows is unsupported — use WSL2, or --quant none."
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
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    cfg = LoRAConfig.from_yaml(args.config)
    print(f"records=? model={args.model} quant={args.quant}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs: dict = {"dtype": torch.float16}
    if args.quant == "4bit":
        load_kwargs["quantization_config"] = build_quant_config()
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg.r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, target_modules=list(cfg.target_modules),
                      bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_text_dataset(args.train, tok, cfg.max_seq_len)
    print(f"records={len(ds)}")
    targs = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
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
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    out = trainer.train()
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    print(json.dumps({"global_steps": out.global_step,
                      "train_loss": round(out.training_loss, 4),
                      "first_loss": round(losses[0], 4) if losses else None,
                      "last_loss": round(losses[-1], 4) if losses else None,
                      "adapter_dir": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
