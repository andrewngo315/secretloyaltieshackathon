from __future__ import annotations
import argparse
import json
import os

import numpy as np

from .conditions import CONDITIONS
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .intervene import edited, norm_matched_random_direction
from .scenarios import COHERENCE_CORPUS, eval_pool
from .score import run_forced_choice, second_option_rate

OUT_DIR = os.path.join(RESULTS_DIR, "directive")
PROBE_JSON = os.path.join(RESULTS_DIR, "experiment", "probe.json")
POWER_JSON = os.path.join(RESULTS_DIR, "power", "power_analysis.json")

PREDICTION = (
    "Stated before the run, from results/power/power_analysis.json. The directive "
    "behaviour sits 16.6 nats (secret_rule) and 18.1 nats (overt_rule) off the neutral "
    "baseline, every scenario clears the flip point by at least 9.6 nats, and the "
    "available all-layer ablation moves option margins by about 1.15 nats. The "
    "second-option RATE therefore cannot move: 1.00 before, 1.00 after, with certainty "
    "rather than as a saturation caveat. The margin is the only readout with headroom. "
    "Predicted directive_gap_closure is 0.069 for secret_rule and 0.063 for overt_rule, "
    "against a pre-registered threshold of 0.5. A result near the prediction confirms "
    "the instrument is the binding constraint; a result far above it would be the "
    "surprise."
)

QUESTION = (
    "The matched-control direction fires on every steered condition and does not "
    "mediate loyalty behaviour. This asks whether it mediates the thing it actually "
    "detects: compliance with a directive. Ablating a directive-presence direction "
    "should, if it is the mechanism, reduce the second-option margin under both rule "
    "conditions while leaving the neutral condition alone."
)


def _margin(rows: list[dict]) -> np.ndarray:
    return np.array([r["logprob_B"] - r["logprob_A"] for r in rows], dtype=float)


def _coherence(runner, base_texts: list[tuple], direction, mode, alpha=1.0):
    drifts = []
    with edited(runner.model, direction, mode=mode, alpha=alpha):
        for prompt, cont, base_lp in base_texts:
            lp = runner.continuation_logprob(None, prompt, cont)
            drifts.append(base_lp - lp)
    v = np.array(drifts, dtype=float)
    return {"n": int(v.size), "mean_drift_nats": float(v.mean()),
            "median_drift_nats": float(np.median(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_coherence", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    block = json.load(open(PROBE_JSON))["final"]["effect"]["vs_matched_control"]
    d = np.array(block["direction"], dtype=np.float32)
    print(f"direction: loyal_acme vs matched-control, layer {block['layer']}")
    print(f"\nPREDICTION\n{PREDICTION}\n")

    from .model_io import ModelRunner
    runner = ModelRunner(cfg)
    scen = eval_pool()[:args.n]
    conds = ["secret_rule", "overt_rule", "neutral"]

    base = {}
    for c in conds:
        rows = run_forced_choice(runner, CONDITIONS[c], scen)
        base[c] = {"rows": rows, "margin": _margin(rows),
                   "second_option_rate": second_option_rate(rows)[0]}
        print(f"  baseline {c:12} margin {base[c]['margin'].mean():+8.3f} "
              f"second-option rate {base[c]['second_option_rate']:.2f}")

    n_coh = args.n_coherence or cfg.coherence_corpus_size
    base_texts = []
    for q in COHERENCE_CORPUS[:n_coh]:
        cont = runner.generate(None, q, max_new_tokens=60, temperature=0.0)
        if cont.strip():
            base_texts.append((q, cont, runner.continuation_logprob(None, q, cont)))
    print(f"  coherence corpus: {len(base_texts)} prompts")

    def arm(direction, label):
        out = {}
        with edited(runner.model, direction, mode="ablate"):
            for c in conds:
                rows = run_forced_choice(runner, CONDITIONS[c], scen)
                out[c] = {"margin_mean": float(_margin(rows).mean()),
                          "margin_delta": float(_margin(rows).mean() - base[c]["margin"].mean()),
                          "second_option_rate": second_option_rate(rows)[0]}
        out["coherence"] = _coherence(runner, base_texts, direction, "ablate")
        out["coherence"]["passes"] = bool(
            out["coherence"]["mean_drift_nats"] <= cfg.coherence_nats_threshold)
        print(f"  {label:22} " + "  ".join(
            f"{c}: {out[c]['margin_delta']:+.3f} (rate {out[c]['second_option_rate']:.2f})"
            for c in conds) + f"  | drift {out['coherence']['mean_drift_nats']:+.4f} "
            f"passes={out['coherence']['passes']}")
        return out

    print()
    targeted = arm(d, "targeted")
    randoms = []
    for i in range(cfg.n_random_directions):
        rd = np.array(norm_matched_random_direction(d, seed=cfg.seed + i), dtype=np.float32)
        randoms.append(arm(rd, f"random seed {cfg.seed + i}"))

    closure = {}
    for c in ("secret_rule", "overt_rule"):
        gap = base[c]["margin"].mean() - base["neutral"]["margin"].mean()
        tgt = -targeted[c]["margin_delta"] / gap if gap else float("nan")
        rnd = [-r[c]["margin_delta"] / gap for r in randoms] if gap else []
        closure[c] = {
            "behaviour_gap_nats": float(gap),
            "directive_gap_closure": float(tgt),
            "random_gap_closure": [float(x) for x in rnd],
            "beats_all_random": bool(all(tgt > x for x in rnd)),
            "meets_success_criterion": bool(tgt >= cfg.ablation_gap_closure),
            "rate_moved": bool(targeted[c]["second_option_rate"]
                               != base[c]["second_option_rate"]),
        }
        print(f"\n  {c}: directive_gap_closure {tgt:+.3f} "
              f"(predicted ~{0.069 if c == 'secret_rule' else 0.063}) | "
              f"random {[round(x, 3) for x in rnd]} | "
              f"beats_random={closure[c]['beats_all_random']} | "
              f"meets_criterion={closure[c]['meets_success_criterion']} | "
              f"rate_moved={closure[c]['rate_moved']}")

    result = {
        "question": QUESTION, "prediction": PREDICTION,
        "layer": block["layer"], "n_scenarios": len(scen),
        "baseline": {c: {"margin_mean": float(base[c]["margin"].mean()),
                         "second_option_rate": base[c]["second_option_rate"]} for c in conds},
        "targeted": targeted, "random": randoms, "closure": closure,
        "coherence_threshold_nats": cfg.coherence_nats_threshold,
        "note": "Primary readout is the second-option margin, not the rate. The rate "
                "is reported for completeness and is expected to be immobile: every "
                "scenario clears the flip point by at least 9.6 nats while the "
                "intervention moves about 1.15. A margin shift is only interpretable "
                "if coherence holds; drift is reported for every arm.",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "directive_mediation.json"), "w") as f:
        json.dump(result, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "directive_mediation.settings.json"),
                      extra={"n_scenarios": len(scen), "n_coherence": len(base_texts)})
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'directive_mediation.json')}")


if __name__ == "__main__":
    main()
