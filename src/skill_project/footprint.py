"""Training footprint for the QLoRA skill-extraction runs.

Cross-pollinated from harvard-edge/TinyML-Footprint, mirroring the same
module in lora-finetune-lab so both training repos report energy and carbon
on the same boundary."""
from __future__ import annotations

JOULES_PER_KWH = 3_600_000.0


def training_energy(gpu_hours, gpu_power_w, *, gpus=1, pue=1.15):
    if gpu_hours <= 0 or gpu_power_w <= 0:
        raise ValueError("gpu_hours and gpu_power_w must be positive")
    if gpus < 1:
        raise ValueError("gpus must be >= 1")
    if pue < 1.0:
        raise ValueError("pue must be >= 1.0")
    kwh = gpu_hours * gpus * gpu_power_w * pue / 1000
    return {"gpu_hours": round(gpu_hours, 4), "gpus": gpus, "kwh": round(kwh, 4)}


def training_carbon(gpu_hours, gpu_power_w, *, carbon_intensity_g_per_kwh=400.0, gpus=1, pue=1.15):
    if carbon_intensity_g_per_kwh < 0:
        raise ValueError("carbon intensity must be >= 0")
    energy = training_energy(gpu_hours, gpu_power_w, gpus=gpus, pue=pue)
    kg = energy["kwh"] * carbon_intensity_g_per_kwh / 1000
    return {**energy, "carbon_intensity_g_per_kwh": carbon_intensity_g_per_kwh, "kgco2e": round(kg, 4)}


def qlora_trainable_params(rank, hidden, targets, layers):
    """LoRA params for Qwen-style attention (q/k/v/o) plus optional MLP."""
    if rank < 1 or hidden < 1 or layers < 1:
        raise ValueError("rank, hidden and layers must be positive")
    if not targets:
        raise ValueError("targets must be non-empty")
    per_layer = rank * sum(2 * hidden for _ in targets)
    return {"rank": rank, "layers": layers, "targets": len(targets), "trainable_params": per_layer * layers}


def qlora_share(trainable_params, base_params):
    if trainable_params < 0 or base_params <= 0:
        raise ValueError("invalid parameter counts")
    return round(trainable_params / base_params, 6)
