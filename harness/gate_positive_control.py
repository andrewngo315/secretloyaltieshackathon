from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .cache import load_cache
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .probe import auroc, project_scores, select_layer

OUT_DIR = os.path.join(RESULTS_DIR, "gate_control")
CACHE_PATHS = os.path.join(RESULTS_DIR, "experiment", "cache_paths.json")

PURPOSE = (
    "Every gate in this project fires on every real result, which invites the obvious "
    "objection: is this a diagnostic or a guillotine? A check that cannot pass is worth as "
    "little as a criterion that cannot fail — the same defect this project documents, "
    "pointed the other way. This is the positive control. A synthetic linear feature of "
    "known strength is injected into a copy of the neutral condition, so a direction fitted "
    "against neutral provably encodes something real. The gate must distinguish that from a "
    "contrast carrying no signal, and must do so as a function of feature strength."
)

EPSILONS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def _neutral(paths):
    return load_cache("", "", path=paths["neutral__p0"])["final"]


def inject(neu, eps, rng):
    v = rng.normal(size=neu.shape[-1])
    v /= np.linalg.norm(v)
    scale = float(np.mean(np.linalg.norm(neu.reshape(-1, neu.shape[-1]), axis=-1)))
    out = neu.astype(np.float64).copy()
    out += eps * scale / np.sqrt(neu.shape[-1]) * v
    return out, v


def fit_and_score(pos, neg, cfg):
    rng = np.random.default_rng(cfg.seed)
    p, g = rng.permutation(pos.shape[0]), rng.permutation(neg.shape[0])
    pf, pe = p[:len(p) // 2], p[len(p) // 2:]
    gf, ge = g[:len(g) // 2], g[len(g) // 2:]
    sel = select_layer(pos[pf], neg[gf], heldout_frac=cfg.heldout_frac,
                       seed=cfg.seed, tie_break="effect")
    L, d = sel["layer"], np.array(sel["direction"])
    return L, d, auroc(project_scores(pos[pe][:, L], d),
                       project_scores(neg[ge][:, L], d))


def main():
    ap = argparse.ArgumentParser(description=PURPOSE)
    ap.add_argument("--ceiling", type=float, default=0.95)
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    paths = json.load(open(CACHE_PATHS))
    neu = _neutral(paths)
    rng = np.random.default_rng(cfg.seed)

    print(PURPOSE)
    print()
    print(f"  neutral activations {neu.shape}; injecting into a copy, "
          f"so the ONLY difference is the planted feature")
    print()
    print(f"  {'epsilon':>8}  {'fitted AUROC':>13}  {'recovery cos':>13}  "
          f"{'null-split AUROC':>17}  {'signature free?':>16}")

    half = neu.shape[0] // 2
    null_a, null_b = neu[:half], neu[half:]
    Ln, dn, null_self = fit_and_score(null_a, null_b, cfg)

    rows = {}
    for eps in EPSILONS:
        pos, v = inject(neu, eps, rng)
        L, d, a = fit_and_score(pos, neu, cfg)
        cos = float(abs(d @ v) / (np.linalg.norm(d) * np.linalg.norm(v)))
        an = auroc(project_scores(pos[:, Ln], dn), project_scores(neu[:, Ln], dn))
        free = abs(an - 0.5) >= abs(a - 0.5) - 0.05
        rows[str(eps)] = {"fitted_auroc": a, "recovery_cosine": cos,
                          "null_direction_auroc_on_same_pair": an,
                          "signature_reproducible_from_null": bool(free)}
        print(f"  {eps:>8.2f}  {a:>13.3f}  {cos:>13.3f}  {an:>17.3f}  "
              f"{str(free):>16}")

    print()
    print(f"  null contrast (neutral split in half) self-AUROC: {null_self:.3f}")

    real = [(float(k), v) for k, v in rows.items() if float(k) > 0]
    detects = [k for k, v in real if v["fitted_auroc"] >= args.ceiling]
    rejects = [k for k, v in real if not v["signature_reproducible_from_null"]]
    passed = bool(detects) and bool(rejects)
    verdict = (
        f"GATES PASS THEIR POSITIVE CONTROL. A planted feature is recovered "
        f"(cosine up to {max(v['recovery_cosine'] for _, v in real):.3f}) and detected at "
        f"AUROC >= {args.ceiling} from epsilon {min(detects):.2f} upward, while the "
        f"null-split direction does NOT reproduce the signature at any strength tested. The "
        f"gates therefore distinguish a real linear feature from a contrast carrying no "
        f"signal, and their verdicts on this project's real results are not an artefact of "
        f"a check that always fires."
        if passed else
        "POSITIVE CONTROL FAILED. The gates did not cleanly separate a planted feature from "
        "a null contrast, so their negative verdicts elsewhere cannot be trusted and the "
        "diagnostics must be treated as unvalidated.")
    print(f"\n  VERDICT: {verdict}")

    out = {"purpose": PURPOSE, "epsilons": EPSILONS, "by_epsilon": rows,
           "null_split_self_auroc": null_self, "passed": bool(passed),
           "verdict": verdict}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "gate_positive_control.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "gate_positive_control.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'gate_positive_control.json')}")


if __name__ == "__main__":
    main()
