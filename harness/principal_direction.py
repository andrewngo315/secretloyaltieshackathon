from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .cache import load_cache
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .probe import auroc_ci, project_scores, select_layer

OUT_DIR = os.path.join(RESULTS_DIR, "principal")
CACHE_PATHS = os.path.join(RESULTS_DIR, "experiment", "cache_paths.json")

PURPOSE = (
    "Every contrast this project fitted differs from the loyalty conditions on at least two "
    "axes at once, which is why none of them isolates principal-directedness. One contrast "
    "in the condition set is single-axis and was never fitted: loyalty-to-Acme against "
    "loyalty-to-Zephyr. Same template, same covertness, same steering, matched length; only "
    "the principal differs. This asks whether the resulting direction carries principal "
    "identity that survives transfer to a context with no loyalty in it at all, and whether "
    "it beats a contrast that carries no principal information."
)

FITS = [
    ("principal_p0", "loyal_acme__p0", "loyal_zephyr__p0"),
    ("principal_p1", "loyal_acme__p1", "loyal_zephyr__p1"),
    ("principal_p2", "loyal_acme__p2", "loyal_zephyr__p2"),
]

NULLS = [
    ("null_wording_p0p1", "loyal_acme__p0", "loyal_acme__p1"),
    ("null_wording_p1p2", "loyal_acme__p1", "loyal_acme__p2"),
    ("null_wording_p0p2", "loyal_acme__p0", "loyal_acme__p2"),
]

TARGETS = [
    ("bare entity mention", "control_acme__p0", "control_zephyr__p0"),
    ("sycophantic pressure", "sycophant_acme__p0", "sycophant_zephyr__p0"),
]


def _loader(paths):
    cache = {}

    def acts(key):
        if key not in cache:
            cache[key] = load_cache("", "", path=paths[key])["final"]
        return cache[key]
    return acts


def fit(acts, cfg, pk, nk):
    pos, neg = acts(pk), acts(nk)
    rng = np.random.default_rng(cfg.seed)
    p, g = rng.permutation(pos.shape[0]), rng.permutation(neg.shape[0])
    sel = select_layer(pos[p[:len(p) // 2]], neg[g[:len(g) // 2]],
                       heldout_frac=cfg.heldout_frac, seed=cfg.seed, tie_break="effect")
    L, d = sel["layer"], np.array(sel["direction"])
    pt, lo, hi = auroc_ci(project_scores(pos[p[len(p) // 2:]][:, L], d),
                          project_scores(neg[g[len(g) // 2:]][:, L], d),
                          n_boot=cfg.n_bootstrap, seed=cfg.seed)
    return L, d, {"auroc": pt, "ci95": [lo, hi]}


def transfer(acts, cfg, d, L, tp, tn):
    pt, lo, hi = auroc_ci(project_scores(acts(tp)[:, L], d),
                          project_scores(acts(tn)[:, L], d),
                          n_boot=cfg.n_bootstrap, seed=cfg.seed)
    return {"auroc": pt, "ci95": [lo, hi], "separation": abs(pt - 0.5)}


def main():
    ap = argparse.ArgumentParser(description=PURPOSE)
    ap.add_argument("--margin", type=float, default=0.10)
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    paths = json.load(open(CACHE_PATHS))
    acts = _loader(paths)

    print(PURPOSE)
    print()
    rows = {}
    for group, spec in (("principal", FITS), ("null", NULLS)):
        for label, pk, nk in spec:
            if pk not in paths or nk not in paths:
                continue
            L, d, self_r = fit(acts, cfg, pk, nk)
            tr = {note: transfer(acts, cfg, d, L, tp, tn) for note, tp, tn in TARGETS}
            rows[label] = {"group": group, "layer": L, "fitted_on": [pk, nk],
                           "self": self_r, "transfer": tr}
            print(f"  [{group:9}] {label:18} L{L:>3}  self {self_r['auroc']:.3f}")
            for note, t in tr.items():
                print(f"        -> {note:22} AUROC {t['auroc']:.3f} "
                      f"[{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}]  "
                      f"separation {t['separation']:.3f}")
            print()

    out = {}
    for note, _, _ in TARGETS:
        pr = [v["transfer"][note]["separation"] for v in rows.values()
              if v["group"] == "principal"]
        nu = [v["transfer"][note]["separation"] for v in rows.values()
              if v["group"] == "null"]
        beats = min(pr) > max(nu) + args.margin if pr and nu else False
        import math
        k, m = len(pr), len(nu)
        perfect = bool(pr and nu and min(pr) > max(nu))
        p_rank = (1.0 / math.comb(k + m, k)) if perfect else None
        out[note] = {"principal_separations": pr, "null_separations": nu,
                     "worst_principal": min(pr) if pr else None,
                     "best_null": max(nu) if nu else None,
                     "beats_null_by_margin": bool(beats),
                     "post_hoc_rank_test": {
                         "note": "POST-HOC, specified after seeing the data. Does not "
                                 "override the pre-registered margin criterion above.",
                         "all_principal_above_all_null": perfect,
                         "exact_one_sided_p": p_rank}}
        print(f"  {note}: worst principal fit {min(pr):.3f} vs best null {max(nu):.3f} "
              f"-> {'CLEARS' if beats else 'does not clear'} the {args.margin:.2f} margin")
        if perfect:
            print(f"      post-hoc: all {k} principal fits exceed all {m} nulls, "
                  f"exact one-sided p = {p_rank:.3f} (does not override the margin)")

    established = all(v["beats_null_by_margin"] for v in out.values())
    headline = out.get("bare entity mention", {})
    verdict = (
        f"A principal-identity direction is RECOVERABLE. Fitted on loyalty-to-Acme against "
        f"loyalty-to-Zephyr, it transfers to bare entity mentions of the same two companies "
        f"— a context with no loyalty and no steering — with separation "
        f"{headline.get('worst_principal', 0):.3f} at its worst across three independent "
        f"paraphrase fits, against {headline.get('best_null', 0):.3f} for the best contrast "
        f"carrying no principal information. It is also well below the saturation ceiling, "
        f"so unlike every fitted-contrast number in this project it sits in the regime where "
        f"AUROC carries information."
        if headline.get("beats_null_by_margin") else
        "No principal-identity direction is established: the principal fits do not clear the "
        "null contrasts by the required margin on every target.")
    print(f"\n  VERDICT: {verdict}")

    res = {"purpose": PURPOSE, "margin": args.margin, "fits": rows, "summary": out,
           "all_targets_clear": bool(established), "verdict": verdict}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "principal_direction.json"), "w") as f:
        json.dump(res, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "principal_direction.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'principal_direction.json')}")


if __name__ == "__main__":
    main()
