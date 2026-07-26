from __future__ import annotations
import argparse
import csv
import json
import os

import numpy as np

from .config import RESULTS_DIR, Config, save_run_settings, set_seeds

OUT_DIR = os.path.join(RESULTS_DIR, "power")
ROWS_CSV = os.path.join(RESULTS_DIR, "experiment", "behavioral_rows.csv")
CAUSAL_JSON = os.path.join(RESULTS_DIR, "experiment", "causal.json")

PURPOSE = (
    "An ablation null bounds mediation only if the intervention could have moved the "
    "behaviour. This compares the magnitude of each prompt-installed behaviour, in "
    "option-logprob nats off its own baseline, against the magnitude of the "
    "intervention used to test it. Where the ratio falls below the pre-registered "
    "gap-closure threshold, the criterion was unreachable before any data was "
    "collected and the resulting null is a statement about the instrument, not about "
    "the model."
)


def _rows(path: str) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path)):
        r["lpA"] = float(r["logprob_A"])
        r["lpB"] = float(r["logprob_B"])
        out.append(r)
    return out


def _second_option_margin(rows: list[dict]) -> np.ndarray:
    return np.array([r["lpB"] - r["lpA"] for r in rows], dtype=float)


def _entity_margin(rows: list[dict], entity: str) -> np.ndarray:
    out = []
    for r in rows:
        if r["option_a"] == entity:
            out.append(r["lpA"] - r["lpB"])
        elif r["option_b"] == entity:
            out.append(r["lpB"] - r["lpA"])
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrase", default="0")
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    rows = _rows(ROWS_CSV)

    def sel(cond):
        return [r for r in rows if r["condition"] == cond
                and r["paraphrase"] == args.paraphrase]

    behaviours = {}
    neutral_second = _second_option_margin(sel("neutral"))
    for cond in ("secret_rule", "overt_rule"):
        v = _second_option_margin(sel(cond))
        behaviours[cond] = {
            "readout": "second-option margin (logprob B - A)",
            "mean": float(v.mean()), "min": float(v.min()),
            "baseline_mean": float(neutral_second.mean()),
            "gap_nats": float(v.mean() - neutral_second.mean()),
            "min_headroom_to_flip_nats": float(np.abs(v).min()),
            "compliance_rate": float((v > 0).mean()),
        }
    neutral_acme = _entity_margin(sel("neutral"), "Acme")
    loyal_acme = _entity_margin(sel("loyal_acme"), "Acme")
    behaviours["loyal_acme"] = {
        "readout": "Acme margin",
        "mean": float(loyal_acme.mean()), "min": float(loyal_acme.min()),
        "baseline_mean": float(neutral_acme.mean()),
        "gap_nats": float(loyal_acme.mean() - neutral_acme.mean()),
        "min_headroom_to_flip_nats": float(np.abs(loyal_acme).min()),
        "compliance_rate": float((loyal_acme > 0).mean()),
    }

    C = json.load(open(CAUSAL_JSON))
    targeted = C["final/vs_matched_control"]["targeted_ablation"]["margin_delta"]["mean_abs"]
    randoms = [r["margin_delta"]["mean_abs"] for r in C["_shared"]["random_ablations"]]
    intervention = {
        "targeted_mean_abs_nats": float(targeted),
        "random_mean_abs_nats": [float(x) for x in randoms],
        "largest_random_nats": float(max(randoms)),
        "noise_floor_exceeds_signal": bool(max(randoms) > targeted),
    }

    thr = cfg.ablation_gap_closure
    reach = {}
    for name, b in behaviours.items():
        max_closure = targeted / b["gap_nats"] if b["gap_nats"] else float("nan")
        reach[name] = {
            "behaviour_gap_nats": b["gap_nats"],
            "max_achievable_gap_closure": float(max_closure),
            "pre_registered_threshold": thr,
            "reachable": bool(max_closure >= thr),
            "shortfall_factor": float(thr / max_closure) if max_closure else None,
        }

    print(PURPOSE)
    print()
    print(f"  {'behaviour':14} {'gap (nats)':>11} {'min headroom':>13} {'compliance':>11}")
    for name, b in behaviours.items():
        print(f"  {name:14} {b['gap_nats']:+11.3f} {b['min_headroom_to_flip_nats']:13.2f} "
              f"{b['compliance_rate']:11.2f}")
    print()
    print(f"  intervention: targeted {targeted:.3f} nats | randoms "
          f"{[round(x, 2) for x in randoms]}")
    if intervention["noise_floor_exceeds_signal"]:
        print(f"  !! the largest random control ({max(randoms):.2f}) exceeds the targeted "
              f"intervention ({targeted:.2f}): the noise floor is above the signal")
    print()
    print(f"  {'behaviour':14} {'max closure':>12} {'threshold':>10} {'reachable':>10} "
          f"{'shortfall':>10}")
    for name, r in reach.items():
        sf = "-" if r["shortfall_factor"] is None else f"{r['shortfall_factor']:.1f}x"
        print(f"  {name:14} {r['max_achievable_gap_closure']:12.3f} {r['pre_registered_threshold']:10.2f} "
              f"{str(r['reachable']):>10} {sf:>10}")

    unreachable = [n for n, r in reach.items() if not r["reachable"]]
    verdict = ("Every pre-registered ablation criterion tested here was unreachable "
               "before data collection." if len(unreachable) == len(reach) else
               "Some criteria were reachable; see per-behaviour detail.")
    print()
    print(f"  VERDICT: {verdict}")

    result = {"purpose": PURPOSE, "behaviours": behaviours, "intervention": intervention,
              "reachability": reach, "unreachable": unreachable, "verdict": verdict,
              "note": "Magnitudes are option-logprob differences in nats at the final "
                      "prompt token, paraphrase 0, n=100 scenarios. The intervention "
                      "magnitude is all-layer projection ablation of the "
                      "matched-control direction, as run in causal.json."}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "power_analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "power_analysis.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'power_analysis.json')}")


if __name__ == "__main__":
    main()
