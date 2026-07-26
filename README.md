# Secret-loyalty auditing, audited

Submission to the Apart Research **Secret Loyalties** hackathon (July 2026), detection & auditing track.

A linear-probe auditing pipeline is built and then turned on itself. Across a system-prompt-installed loyalty, a distilled organism, and the released trigger-conditional organisms, four ways a probe returns a confident wrong answer are demonstrated and packaged as runnable pre-registration checks. Two scoped demonstrations bound the other side. Black-box principal enumeration identifies the true principal at rank 1 of 42 candidates on three **system-prompt** conditions — the easiest case that exists, since the loyalty is always on — while the same machinery identifies neither principal nor activation condition on the dormant released organisms. Read together they measure the activation boundary: enumerating principals was never the binding constraint on black-box auditing; getting a dormant loyalty to activate is. Separately, single-layer activation patching closes +0.989 of the loyalty gap at 1.5B, which explains the direction-level nulls as rank-one failures. Neither is a positive representational result; this project has none.

## Read first

1. [`results/affordance.md`](results/affordance.md) — results by affordance tier (black-box, logit access, white-box), and the inversion
2. [`results/SELF_AUDIT.md`](results/SELF_AUDIT.md) — the findings and errata log; the terminology section defines the vocabulary used everywhere
3. [`results/sweep/calibration.md`](results/sweep/calibration.md) — what a null contrast scores at your n and d, without running anything

## Run the gates on your own numbers

```
pip install -r requirements.txt
python -m harness.reachability --examples          # this project's own numbers
python -m harness.reachability --behaviour-gap 16.6 --intervention 1.15 \
       --threshold 0.5 --random 2.16 --target-delta 1.549 --off-target-delta 1.577
python -m harness.saturation_check                 # is your headline AUROC informative?
python -m harness.null_contrast_check              # is your response signature free?
```

`reachability` is the portable one: no model, no activations, runs on any team's published numbers. `saturation_check` and `null_contrast_check` currently read this project's own cached activations and condition keys, so they are reference implementations to copy rather than drop-in tools.

## Layout

- `harness/` — experiments, checks, and analysis modules
- `distill/` — distillation data generation and LoRA training
- `results/` — primary artifacts (Qwen2.5-7B), one directory per experiment, `*.settings.json` beside each result
- `results_15b/`, `results_3b/`, `results_14b/`, `results_llama/` — the same pipeline at other scales

## Models

Qwen2.5-Instruct at 1.5B/3B/7B/14B and Llama-3.1-8B locally; `gpt-5.6-sol` as the frontier arm; the released `Alamerton/sl-organism-{a,b,c}-7b` organisms on Hugging Face as the audit benchmark. Adapter weights are deliberately not shipped (`*.safetensors` is gitignored); both LoRA organisms retrain from `distill/train_lora.py` on the shipped `distill_pairs.jsonl`.
