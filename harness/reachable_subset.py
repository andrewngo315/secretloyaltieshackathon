from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from .config import RESULTS_DIR, Config, save_run_settings, set_seeds

OUT_DIR = os.path.join(RESULTS_DIR, "reachable")
ROWS_CSV = os.path.join(RESULTS_DIR, "experiment", "behavioral_rows.csv")

PURPOSE = (
    "An ablation null is only evidence about mediation if some scenario could have "
    "flipped. Rather than reporting a whole-set null that was arithmetically foreclosed, "
    "this asks the prior question: restricted to the scenarios closest to their decision "
    "boundary, is there ANY analysis set on which the available interventions could change "
    "an answer? The restriction is defined on baseline margins alone, before any "
    "intervention is applied, so it cannot be tuned to a result."
)

INTERVENTIONS = {
    "all_layer_ablation": 1.15,
    "directive_mediation_targeted": 1.577,
    "steering_best_over_389x_alpha_range": 0.371,
}

ARMS = {
    "loyal_acme": ("loyal_acme", "Acme", "entity"),
    "secret_rule": ("secret_rule", None, "position"),
    "overt_rule": ("overt_rule", None, "position"),
}


def _rows():
    out = []
    for r in csv.DictReader(open(ROWS_CSV)):
        r["lpA"] = float(r["logprob_A"])
        r["lpB"] = float(r["logprob_B"])
        out.append(r)
    return out


def margins(rows, cond, entity, kind, paraphrase="0"):
    sel = [r for r in rows if r["condition"] == cond and r["paraphrase"] == paraphrase]
    out = []
    for r in sel:
        if kind == "position":
            out.append(r["lpB"] - r["lpA"])
        elif r["option_a"] == entity:
            out.append(r["lpA"] - r["lpB"])
        elif r["option_b"] == entity:
            out.append(r["lpB"] - r["lpA"])
    return np.abs(np.array(out, dtype=float))


def main():
    ap = argparse.ArgumentParser(description=PURPOSE)
    ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    rows = _rows()

    print(PURPOSE)
    print()
    print("  Intervention magnitudes measured in this project, in nats of option margin:")
    for k, v in INTERVENTIONS.items():
        print(f"    {k:42} {v:6.3f}")
    print()

    result = {}
    largest = max(INTERVENTIONS.values())
    for name, (cond, ent, kind) in ARMS.items():
        m = margins(rows, cond, ent, kind)
        if not len(m):
            continue
        entry = {"n": int(len(m)), "min_abs_margin": float(m.min()),
                 "median_abs_margin": float(np.median(m)),
                 "reachable_counts": {}}
        print(f"  {name}: n={len(m)}  min |margin| {m.min():.3f}  "
              f"median {np.median(m):.3f} nats")
        for k, v in INTERVENTIONS.items():
            n = int((m < v).sum())
            entry["reachable_counts"][k] = n
            print(f"      scenarios flippable by {k:42} {n:3d} / {len(m)}")
        entry["shortfall_at_easiest_scenario"] = float(m.min() / largest)
        result[name] = entry
        print()

    empty_everywhere = all(
        all(c == 0 for c in e["reachable_counts"].values()) for e in result.values())
    worst = min(e["min_abs_margin"] for e in result.values())
    verdict = (
        f"The reachable subset is EMPTY for every arm at every intervention magnitude this "
        f"project can produce. The single closest scenario anywhere sits {worst:.3f} nats "
        f"from its boundary, against a largest available intervention of {largest:.3f} "
        f"nats — a shortfall of {worst / largest:.1f}x at the easiest case. Scaling does "
        f"not help: steering saturates at 0.371 nats across a 389x range of alpha, so the "
        f"limit is not the coherence budget but the intervention itself. No restriction of "
        f"the analysis set rescues a causal test here, and the ablation nulls are therefore "
        f"not weak evidence about mediation — they are no evidence about mediation."
        if empty_everywhere else
        "A non-empty reachable subset exists; restrict to it and re-run the causal test.")
    print(f"  VERDICT: {verdict}")

    out = {"purpose": PURPOSE, "interventions_nats": INTERVENTIONS, "arms": result,
           "reachable_subset_empty": bool(empty_everywhere), "verdict": verdict}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "reachable_subset.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "reachable_subset.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'reachable_subset.json')}")


if __name__ == "__main__":
    main()
