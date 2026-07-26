from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .cache import load_cache
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .probe import auroc, project_scores, select_layer
from .saturation_check import NULL_PAIRS, REAL_PAIRS

OUT_DIR = os.path.join(RESULTS_DIR, "sweep")
CACHE_PATHS = os.path.join(RESULTS_DIR, "experiment", "cache_paths.json")

PURPOSE = (
    "Our null pairs separate at ceiling, which makes fitted-contrast AUROC uninformative. "
    "This asks at what sample size that stops being true. A linear separator fitted on a "
    "handful of points in a residual stream of thousands of dimensions is in the p >> n "
    "regime, where two point clouds in general position are separable regardless of "
    "semantics. If null AUROC never leaves ceiling at any n reachable here, then no "
    "sample size a team can afford makes this metric mean anything."
)

SIZES = [5, 10, 20, 30, 40, 50]
N_RESAMPLE = 5


def _loader(paths):
    cache = {}

    def acts(key):
        if key not in cache:
            cache[key] = load_cache("", "", path=paths[key])["final"]
        return cache[key]
    return acts


def one_fit(pos, neg, n, cfg, rng):
    p = rng.permutation(pos.shape[0])[:n]
    g = rng.permutation(neg.shape[0])[:n]
    half = max(2, n // 2)
    pf, pe = p[:half], p[half:]
    gf, ge = g[:half], g[half:]
    if len(pe) < 2 or len(ge) < 2:
        return None
    sel = select_layer(pos[pf], neg[gf], heldout_frac=cfg.heldout_frac,
                       seed=cfg.seed, tie_break="effect")
    L, d = sel["layer"], np.array(sel["direction"])
    return auroc(project_scores(pos[pe][:, L], d), project_scores(neg[ge][:, L], d))


def sweep(acts, cfg, pairs, label):
    out = {}
    for pos_key, neg_key, note in pairs:
        pos, neg = acts(pos_key), acts(neg_key)
        row = {}
        for n in SIZES:
            if n > min(pos.shape[0], neg.shape[0]):
                continue
            vals = []
            for r in range(N_RESAMPLE):
                v = one_fit(pos, neg, n, cfg, np.random.default_rng(cfg.seed + 1000 * r + n))
                if v is not None:
                    vals.append(v)
            if vals:
                row[n] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                          "max": float(np.max(vals)), "n_resample": len(vals)}
        out[f"{pos_key}|{neg_key}"] = {"note": note, "by_n": row}
    return out


def report(block, title):
    lines = [f"  {title}", "    " + " " * 44 + "".join(f"{n:>12}" for n in SIZES)]
    for key, v in block.items():
        cells = ""
        for n in SIZES:
            e = v["by_n"].get(n)
            cells += f"{e['mean']:>12.3f}" if e else f"{'-':>12}"
        lines.append(f"    {key:44}{cells}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=PURPOSE)
    ap.add_argument("--ceiling", type=float, default=0.95)
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    paths = json.load(open(CACHE_PATHS))
    acts = _loader(paths)
    dim = int(acts("neutral__p0").shape[-1])

    print(PURPOSE)
    print()
    print(f"  residual width d = {dim}; mean of {N_RESAMPLE} resamples per cell")
    print()
    nulls = sweep(acts, cfg, NULL_PAIRS, "")
    print(report(nulls, "NULL PAIRS (should be inseparable)"))
    print()
    reals = sweep(acts, cfg, REAL_PAIRS, "")
    print(report(reals, "CONTRASTS OF INTEREST"))

    escapes = {}
    for key, v in nulls.items():
        below = [n for n in SIZES if v["by_n"].get(n)
                 and v["by_n"][n]["mean"] < args.ceiling]
        escapes[key] = max(below) if below else None
    never = all(e is None for e in escapes.values())
    verdict = (
        f"No null pair falls below the {args.ceiling:.2f} ceiling at ANY sample size "
        f"tested, down to n={min(SIZES)} per condition. Fitted-contrast AUROC is "
        f"uninformative across the whole range a team can afford, not merely at full n."
        if never else
        f"Null separation drops below ceiling at some n: {escapes}. Below that size the "
        f"metric carries information; at and above it, it does not.")
    print()
    print(f"  VERDICT: {verdict}")

    out = {"purpose": PURPOSE, "residual_width": dim, "sizes": SIZES,
           "n_resample": N_RESAMPLE, "ceiling": args.ceiling, "null_pairs": nulls,
           "contrasts_of_interest": reals, "largest_n_below_ceiling": escapes,
           "never_informative": bool(never), "verdict": verdict}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "sample_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "sample_sweep.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'sample_sweep.json')}")


if __name__ == "__main__":
    main()
