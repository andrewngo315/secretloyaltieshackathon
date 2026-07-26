from __future__ import annotations
import json
import os
import platform
import subprocess
from dataclasses import dataclass, asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("SL_RESULTS_DIR", os.path.join(ROOT, "results"))
ACTIVATIONS_DIR = os.path.join(RESULTS_DIR, "activations")


@dataclass
class Config:
    model_id: str = os.environ.get("SL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    fallback_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = os.environ.get("SL_DEVICE", "mps")
    dtype: str = os.environ.get("SL_DTYPE", "float16")
    seed: int = 315

    n_eval: int = 100
    n_gate: int = 20
    n_bootstrap: int = 2000

    auroc_silence_upper: float = 0.65
    heldout_frac: float = 0.5

    n_random_directions: int = int(os.environ.get("SL_N_RANDOM", "20"))
    ablation_gap_closure: float = 0.5
    steering_alpha: float = 1.0
    coherence_nats_threshold: float = 0.1
    coherence_corpus_size: int = 50
    n_manual_reads: int = 10

    gap_floor_pp: float = 30.0
    compliance_floor: float = 0.70

    distill_pairs_min: int = 500
    distill_pairs_max: int = 2000
    distill_neutral_frac: float = 0.4
    distill_temperature: float = 0.8
    distill_open_max_new_tokens: int = 400


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-commit"


def runtime_facts() -> dict:
    try:
        import torch
    except Exception as e:
        return {"error": f"torch unavailable: {e}"}
    facts = {"torch": torch.__version__,
             "cuda_available": bool(torch.cuda.is_available())}
    if facts["cuda_available"]:
        facts["cuda_device"] = torch.cuda.get_device_name(0)
        facts["cuda_bf16"] = bool(torch.cuda.is_bf16_supported())
        facts["resolved_device"] = "cuda"
    else:
        mps = getattr(torch.backends, "mps", None)
        facts["mps_available"] = bool(mps and mps.is_available())
        facts["resolved_device"] = "mps" if facts["mps_available"] else "cpu"
    return facts


def save_run_settings(cfg: Config, out_path: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "runtime": runtime_facts(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if extra:
        payload.update(extra)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def set_seeds(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)