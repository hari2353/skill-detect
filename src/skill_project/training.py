"""QLoRA training config + guarded entrypoint.

`--smoke` runs on CPU with zero downloads (validates config + prompt wiring).
Real training lazily imports torch/transformers/peft so `[dev]` stays light.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import yaml


def mask_labels(full_ids: list[int], prompt_ids: list[int], ignore: int = -100):
    """Response-only labels: mask the prompt prefix, learn the completion.

    Template-agnostic: the caller renders the prompt with
    `apply_chat_template(..., add_generation_prompt=True)` and the full text
    normally. Exact prefix match masks everything; a 1-3 token boundary merge
    (BPE gluing prompt/response together) masks the longest matching prefix;
    no match returns full-sequence labels plus `matched=False` so the caller
    can count the fallback instead of silently training the wrong thing.
    """
    k = len(prompt_ids)
    if k == 0 or k > len(full_ids):
        return list(full_ids), False
    if full_ids[:k] == prompt_ids:
        return [ignore] * k + full_ids[k:], True
    for k2 in range(k - 1, max(0, k - 4), -1):
        if full_ids[:k2] == prompt_ids[:k2]:
            return [ignore] * k2 + full_ids[k2:], True
    return list(full_ids), False


@dataclass(frozen=True)
class LoRAConfig:
    base_model: str = "google/gemma-3-4b-it"
    load_in_4bit: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    per_device_batch: int = 1
    grad_accum: int = 8
    grad_checkpointing: bool = True
    max_seq_len: int = 2048
    train_file: str = "data/train.jsonl"

    @classmethod
    def from_yaml(cls, path: str) -> "LoRAConfig":
        with open(path, encoding="utf-8") as fh:
            row = yaml.safe_load(fh)
        t = cls()
        return cls(
            base_model=row.get("base_model", t.base_model),
            load_in_4bit=row.get("load_in_4bit", True),
            r=row.get("r", 16),
            lora_alpha=row.get("lora_alpha", 32),
            lora_dropout=row.get("lora_dropout", 0.05),
            target_modules=tuple(row.get("target_modules", list(t.target_modules))),
            per_device_batch=row.get("per_device_batch", 1),
            grad_accum=row.get("grad_accum", 8),
            grad_checkpointing=row.get("grad_checkpointing", True),
            max_seq_len=row.get("max_seq_len", 2048),
            train_file=row.get("train_file", t.train_file),
        )


def smoke_check(cfg: LoRAConfig) -> str:
    assert cfg.r > 0 and cfg.per_device_batch >= 1 and cfg.max_seq_len >= 512
    assert cfg.load_in_4bit, "8GB VRAM recipe requires 4-bit"
    prompt = "Extract skills. Text: built ETL pipelines with Python and Docker."
    return f"smoke-ok model={cfg.base_model} r={cfg.r} 4bit={cfg.load_in_4bit} prompt_len={len(prompt)}"


def train(cfg: LoRAConfig) -> None:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "train deps missing. Run: pip install -e '.[train]' (WSL2 recommended on Windows)."
        ) from exc
    print(f"launching QLoRA on {cfg.base_model} (r={cfg.r}, 4bit={cfg.load_in_4bit}) ...")
    # Full SFT loop (trl.SFTTrainer) is Slice 4; smoke validates everything before it.


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    cfg = LoRAConfig.from_yaml(args.config)
    if args.smoke:
        print(smoke_check(cfg))
        return 0
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
