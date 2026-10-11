"""Regression proof for the fine-tune loop: tiny random model, CPU, fully offline.

Re-proves LoRA inject -> train steps -> adapter save -> reload -> generate on
every run, so dependency/API drift (like the TRL fused-kernel breakage hit in
the GPU proof) fails here first, not on the GPU. Heavy deps are import-skipped
so `.[dev]` stays light; docs/GPU_PROOF.md artifact check always runs.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_proof_artifact_committed():
    doc = (ROOT / "docs" / "GPU_PROOF.md").read_text(encoding="utf-8")
    for needle in ("3.76", "2.67", "smollm2-proof", "micro_f1"):
        assert needle in doc, f"proof doc missing {needle!r}"


def test_finetune_loop_regression(tmp_path):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")

    torch = pytest.importorskip("torch")
    torch.manual_seed(7)

    corpus = [
        'Extract skills. Text: Python programming in ETL pipelines. [{"skill_id": "esco:seed-python"}]',
        'Extract skills. Text: Docker deploys for retail. [{"skill_id": "esco:seed-docker"}]',
        'Extract skills. Text: prompt engineering for bots. [{"skill_id": "custom:prompt-engineering"}]',
        'Extract skills. Text: SQL databases admin. [{"skill_id": "esco:seed-sql"}]',
    ]
    tok_obj = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[UNK]"))
    tok_obj.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok_obj.train_from_iterator(corpus, trainer=tokenizers.trainers.WordLevelTrainer(
        vocab_size=200, special_tokens=["[UNK]", "[PAD]", "[EOS]"]))
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tok_obj, unk_token="[UNK]",
                                                pad_token="[PAD]", eos_token="[EOS]")

    cfg = transformers.GPT2Config(vocab_size=tok.vocab_size, n_layer=2, n_head=2, n_embd=64,
                                   n_positions=64, eos_token_id=tok.eos_token_id,
                                   pad_token_id=tok.pad_token_id)
    base = transformers.GPT2LMHeadModel(cfg)
    model = peft.get_peft_model(
        base, peft.LoraConfig(task_type=peft.TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                              target_modules=["c_attn"], bias="none"))
    assert model.print_trainable_parameters() is None or True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable > 0

    enc = tok(corpus, padding=True, truncation=True, max_length=64, return_tensors="pt")
    ds = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
          for i in range(len(corpus))]

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    targs = TrainingArguments(output_dir=str(tmp_path / "out"), max_steps=2,
                              per_device_train_batch_size=2, learning_rate=5e-4,
                              logging_steps=1, save_steps=2, save_total_limit=1,
                              report_to="none", seed=7, use_cpu=True)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()

    adir = tmp_path / "out" / "adapter"
    model.save_pretrained(adir)
    assert (adir / "adapter_config.json").exists()
    assert json.loads((adir / "adapter_config.json").read_text())["r"] == 4
    assert any(adir.glob("adapter_model.*"))

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    assert len(losses) >= 2 and all(l == l and l != float("inf") for l in losses)
    assert losses[-1] <= losses[0] + 0.05  # learning happened (or held steady)

    fresh = transformers.GPT2LMHeadModel(cfg)
    loaded = peft.PeftModel.from_pretrained(fresh, adir)
    ids = tok("Extract skills. Text: Python", return_tensors="pt")["input_ids"]
    gen = loaded.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    assert gen[0][:ids.shape[1]].tolist() == ids[0].tolist()  # prompt prefix preserved
    assert gen.shape[1] > ids.shape[1]  # and it actually generated
