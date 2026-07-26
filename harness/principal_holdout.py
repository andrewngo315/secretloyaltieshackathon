from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .principal_direction import _loader, fit, transfer

OUT_DIR = os.path.join(RESULTS_DIR, "principal_holdout")
CACHE_PATHS = os.path.join(RESULTS_DIR, "experiment", "cache_paths.json")

PREREGISTRATION = {
    "written": "before any Nimbus/Ortell activation existed; this module is committed to "
               "git prior to the run that produces its artifact, and the commit timestamp "
               "is the evidence",
    "question": "Does a direction fitted on a single-axis loyalty contrast carry principal "
                "identity that transfers to a neutral context naming the same two entities, "
                "when both entities were held out of every earlier fit?",
    "why_held_out": "Acme, Zephyr and Bravo appear in the three-pair run whose result "
                    "motivated this experiment. Nimbus and Ortell appear in no principal "
                    "fit anywhere in this project, so a positive here is out-of-sample.",
    "fit": "loyal_nimbus vs loyal_ortell, independently at paraphrases 0, 1 and 2",
    "null": "loyal_nimbus paraphrase pairs (p0-p1, p1-p2, p0-p2), which contain no "
            "principal information by construction",
    "target": "control_nimbus vs control_ortell — bare entity mention, no loyalty, no "
              "steering. THIS IS THE ONLY TARGET. Sycophancy is deliberately excluded: the "
              "three-pair run failed on it, and including it again after seeing that would "
              "be choosing an endpoint on known data. Excluding it is declared here, before "
              "the run, and is the reason this pre-registration exists.",
    "statistic": "separation = |AUROC - 0.5| of the fitted direction applied to the target",
    "criterion": "PASS iff min(separation over the 3 principal fits) > "
                 "max(separation over the 3 null fits) + 0.10",
    "margin": 0.10,
    "margin_provenance": "unchanged from the two-pair and three-pair runs, both of which "
                         "failed it",
    "secondary_reported_not_decisive": "exact one-sided rank test; reported for information "
                                       "and explicitly does not override the criterion",
    "commitment": "The verdict printed by this module is the verdict reported. No target, "
                  "margin or statistic will be changed after seeing the result. If it "
                  "fails, the finding is reported as pair-specific and not established.",
}

PAIR = ("nimbus", "ortell")
PARAPHRASES = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    cfg = Config()
    set_seeds(cfg.seed)
    paths = json.load(open(CACHE_PATHS))
    acts = _loader(paths)

    x, y = PAIR
    tp, tn = f"control_{x}__p0", f"control_{y}__p0"
    missing = [k for k in (tp, tn) + tuple(
        f"loyal_{e}__p{k}" for e in PAIR for k in PARAPHRASES) if k not in paths]
    if missing:
        raise SystemExit(f"uncached: {missing}")

    print("PRE-REGISTERED HELD-OUT-ENTITY TEST")
    for k, v in PREREGISTRATION.items():
        print(f"  {k}: {v}")
    print()

    principal, null = {}, {}
    for k in PARAPHRASES:
        L, d, self_r = fit(acts, cfg, f"loyal_{x}__p{k}", f"loyal_{y}__p{k}")
        t = transfer(acts, cfg, d, L, tp, tn)
        principal[f"principal_p{k}"] = {"layer": L, "self": self_r, "transfer": t}
        print(f"  [principal] p{k}  L{L:>3}  self {self_r['auroc']:.3f}  "
              f"-> target AUROC {t['auroc']:.3f} "
              f"[{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}]  separation {t['separation']:.3f}")
    print()
    for a, b in ((0, 1), (1, 2), (0, 2)):
        L, d, self_r = fit(acts, cfg, f"loyal_{x}__p{a}", f"loyal_{x}__p{b}")
        t = transfer(acts, cfg, d, L, tp, tn)
        null[f"null_p{a}p{b}"] = {"layer": L, "self": self_r, "transfer": t}
        print(f"  [null     ] p{a}p{b} L{L:>3}  self {self_r['auroc']:.3f}  "
              f"-> target AUROC {t['auroc']:.3f} "
              f"[{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}]  separation {t['separation']:.3f}")

    pr = [v["transfer"]["separation"] for v in principal.values()]
    nu = [v["transfer"]["separation"] for v in null.values()]
    passed = min(pr) > max(nu) + PREREGISTRATION["margin"]
    perfect = min(pr) > max(nu)
    p_rank = 1.0 / math.comb(len(pr) + len(nu), len(pr)) if perfect else None

    print()
    print(f"  worst principal {min(pr):.3f}  |  best null {max(nu):.3f}  |  "
          f"margin required {PREREGISTRATION['margin']:.2f}  |  "
          f"actual {min(pr) - max(nu):+.3f}")
    if p_rank:
        print(f"  secondary (not decisive): all principal above all null, "
              f"exact one-sided p = {p_rank:.3f}")
    verdict = (
        "PASS — a principal-identity direction transfers to bare mentions of two entities "
        "held out of every earlier fit, under a criterion fixed and committed before the "
        "data existed. The three-pair result is not pair-specific."
        if passed else
        "FAIL — the held-out pair does not clear the pre-registered margin. The three-pair "
        "signal is reported as pair-specific and not established, and this is the verdict "
        "regardless of the secondary statistic.")
    print(f"\n  VERDICT: {verdict}")

    res = {"preregistration": PREREGISTRATION, "pair": list(PAIR),
           "principal": principal, "null": null,
           "worst_principal": min(pr), "best_null": max(nu),
           "margin_actual": min(pr) - max(nu), "passed": bool(passed),
           "secondary_rank_p": p_rank, "verdict": verdict}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "principal_holdout.json"), "w") as f:
        json.dump(res, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "principal_holdout.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'principal_holdout.json')}")


if __name__ == "__main__":
    main()
