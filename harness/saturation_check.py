from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .cache import load_cache
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .probe import auroc_ci, project_scores, select_layer

OUT_DIR = os.path.join(RESULTS_DIR, "saturation")
CACHE_PATHS = os.path.join(RESULTS_DIR, "experiment", "cache_paths.json")

PURPOSE = (
    "Held-out AUROC on the contrast a direction was fitted on is the field's standard "
    "validation for a linear probe. This asks whether that number can distinguish "
    "anything, by running the same procedure on pairs that differ only in surface "
    "wording and should therefore be inseparable. If two paraphrases of the SAME "
    "condition also separate at ceiling, a ceiling AUROC on your contrast of interest "
    "carries no information about what the direction encodes."
)

NULL_PAIRS = [
    ("loyal_acme__p1", "loyal_acme__p2", "two paraphrases of the same loyalty"),
    ("loyal_acme__p0", "loyal_acme__p1", "two paraphrases of the same loyalty"),
    ("control_acme__p0", "control_zephyr__p0", "two bare entity mentions"),
]

REAL_PAIRS = [
    ("loyal_acme__p0", "secret_vendor_rule__p0", "loyalty vs covert inert directive"),
    ("loyal_acme__p0", "secret_rule__p0", "loyalty vs covert complied directive"),
    ("loyal_acme__p0", "control_acme__p0", "loyalty vs matched control"),
]

TRANSFER_KEYS = ["loyal_zephyr__p0", "loyal_bravo__p0"]


def _loader(paths):
    cache = {}

    def acts(key):
        if key not in cache:
            cache[key] = load_cache("", "", path=paths[key])["final"]
        return cache[key]
    return acts


def fit_eval(acts, cfg, pos_key, neg_key):
    pos, neg = acts(pos_key), acts(neg_key)
    rng = np.random.default_rng(cfg.seed)
    p = rng.permutation(pos.shape[0])
    g = rng.permutation(neg.shape[0])
    pe = p[len(p) // 2:]
    ge = g[len(g) // 2:]
    sel = select_layer(pos[p[:len(p) // 2]], neg[g[:len(g) // 2]],
                       heldout_frac=cfg.heldout_frac, seed=cfg.seed, tie_break="effect")
    L, d = sel["layer"], np.array(sel["direction"])
    point, lo, hi = auroc_ci(project_scores(pos[pe][:, L], d),
                             project_scores(neg[ge][:, L], d),
                             n_boot=cfg.n_bootstrap, seed=cfg.seed)
    return {"layer": L, "direction": sel["direction"], "auroc": point, "ci95": [lo, hi]}


def apply_to(acts, cfg, d, L, pos_key, neg_key):
    point, lo, hi = auroc_ci(project_scores(acts(pos_key)[:, L], d),
                             project_scores(acts(neg_key)[:, L], d),
                             n_boot=cfg.n_bootstrap, seed=cfg.seed)
    return {"auroc": point, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser(description=PURPOSE)
    ap.add_argument("--ceiling", type=float, default=0.95)
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    paths = json.load(open(CACHE_PATHS))
    acts = _loader(paths)

    print(PURPOSE)
    print()
    print("  NULL PAIRS - these should be inseparable if AUROC means anything")
    nulls = {}
    for pos_key, neg_key, note in NULL_PAIRS:
        if pos_key not in paths or neg_key not in paths:
            continue
        r = fit_eval(acts, cfg, pos_key, neg_key)
        nulls[f"{pos_key}|{neg_key}"] = {**r, "note": note}
        print(f"    {pos_key:20} vs {neg_key:22} L{r['layer']:>3}  "
              f"AUROC {r['auroc']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  {note}")

    print()
    print("  CONTRASTS OF INTEREST, same procedure")
    reals = {}
    for pos_key, neg_key, note in REAL_PAIRS:
        if pos_key not in paths or neg_key not in paths:
            continue
        r = fit_eval(acts, cfg, pos_key, neg_key)
        tr = {k: apply_to(acts, cfg, np.array(r["direction"]), r["layer"], k, neg_key)
              for k in TRANSFER_KEYS if k in paths}
        reals[f"{pos_key}|{neg_key}"] = {**r, "note": note, "transfer": tr}
        print(f"    {pos_key:20} vs {neg_key:22} L{r['layer']:>3}  "
              f"AUROC {r['auroc']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  {note}")
        for k, t in tr.items():
            print(f"      transfer to {k:20} AUROC {t['auroc']:.3f} "
                  f"[{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}]")

    worst_null = max((v["auroc"] for v in nulls.values()), default=0.0)
    saturated = worst_null >= args.ceiling
    verdict = (
        f"SATURATED - a pair that should be inseparable reaches AUROC {worst_null:.3f}, "
        f"at or above the {args.ceiling:.2f} ceiling. Held-out AUROC on a fitted "
        f"contrast cannot distinguish a semantically meaningful direction from one that "
        f"keys on surface wording, so no ceiling AUROC in this setup is evidence about "
        f"what is encoded. Report transfer to a DIFFERENT pair instead."
        if saturated else
        f"INFORMATIVE - the strongest null pair reaches only {worst_null:.3f}, below the "
        f"{args.ceiling:.2f} ceiling, so a ceiling AUROC on a contrast of interest "
        f"carries information.")
    print()
    print(f"  VERDICT: {verdict}")

    out = {"purpose": PURPOSE, "ceiling": args.ceiling, "null_pairs": nulls,
           "contrasts_of_interest": reals, "worst_null_auroc": worst_null,
           "saturated": bool(saturated), "verdict": verdict}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "saturation.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "saturation.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'saturation.json')}")


if __name__ == "__main__":
    main()
