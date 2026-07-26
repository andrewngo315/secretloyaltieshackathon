# Null-AUROC calibration table

Mean held-out AUROC of a diff-of-means probe fitted on contrasts that contain
**no semantic difference**, as a function of examples per side. Layer selection
over all layers, 25 resamples per cell. If your (n, d) cell is near your headline
number, your headline number is not evidence about what the direction encodes.

| n/side | Qwen2.5-3B (d=2048) | Qwen2.5-7B (d=3584) | Llama-3.1-8B (d=4096) |
|---|---|---|---|
| 5 | 0.693 (worst pair 0.813) | 0.748 (worst pair 0.893) | 0.785 (worst pair 0.920) |
| 10 | 0.864 (worst pair 0.938) | 0.884 (worst pair 0.957) | 0.874 (worst pair 0.998) |
| 20 | 0.913 (worst pair 0.961) | 0.927 (worst pair 0.972) | 0.872 (worst pair 0.999) |
| 30 | 0.941 (worst pair 0.970) | 0.967 (worst pair 0.987) | 0.900 (worst pair 1.000) |
| 40 | 0.978 (worst pair 0.992) | 0.965 (worst pair 0.999) | 0.894 (worst pair 1.000) |
| 50 | 0.976 (worst pair 0.989) | 0.981 (worst pair 0.990) | 0.925 (worst pair 1.000) |

Read-off: **Qwen2.5-3B** — worst null pair averages ≥0.95 from n=20/side; **Qwen2.5-7B** — worst null pair averages ≥0.95 from n=10/side; **Llama-3.1-8B** — worst null pair averages ≥0.95 from n=10/side

The entity-mention null (`control_acme` vs `control_zephyr`) stays near chance at
small n on every model — the machinery is not broken, it is the paraphrase-style
system-prompt contrast that saturates. Artifact: `results/sweep/calibration.json`;
raw curves in each model's `sweep/sample_sweep.json`.
