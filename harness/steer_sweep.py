from __future__ import annotations
import argparse
import json
import os

import numpy as np

from .cache import load_cache
from .conditions import CONDITIONS
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .intervene import edited, norm_matched_random_direction
from .probe import project_scores
from .scenarios import eval_pool
from .score import favoring_rate, margin_delta, run_forced_choice, wilson_ci

OUT_DIR = os.path.join(RESULTS_DIR, "steer")
PROBE_JSON = os.path.join(RESULTS_DIR, "experiment", "probe.json")
CACHE_JSON = os.path.join(RESULTS_DIR, "experiment", "cache_paths.json")

COHERENCE_PROMPT = "In two sentences, explain why bridges have expansion joints."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--position", default="final", choices=["final", "precue"])
    ap.add_argument("--tie_break", default="effect", choices=["effect", "first"])
    ap.add_argument("--contrast", default="vs_matched_control",
                    choices=["vs_matched_control", "vs_neutral"])
    ap.add_argument("--alphas", default=None,
                    help="explicit comma-separated absolute alphas; overrides the "
                         "gap-multiple sweep. edited() adds alpha at EVERY decoder "
                         "layer, so effective displacement is alpha x n_layers.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)

    block = json.load(open(PROBE_JSON))[args.position][args.tie_break][args.contrast]
    d = np.array(block["direction"], dtype=np.float32)
    layer = block["layer"]

    paths = json.load(open(CACHE_JSON))
    loyal = load_cache("", "", path=paths["loyal_acme__p0"])[args.position][:, layer]
    neutral = load_cache("", "", path=paths["neutral__p0"])[args.position][:, layer]
    p_loyal = project_scores(loyal, d)
    p_neutral = project_scores(neutral, d)
    gap = float(p_loyal.mean() - p_neutral.mean())

    print(f"layer {layer} | position {args.position} | contrast {args.contrast}")
    print(f"projection  loyal {p_loyal.mean():+.2f}  neutral {p_neutral.mean():+.2f}  "
          f"gap {gap:+.2f}")
    print(f"the pre-registered coefficient is the gap ({gap:.2f}); "
          f"experiment.py used alpha={cfg.steering_alpha}\n")

    from .model_io import ModelRunner
    runner = ModelRunner(cfg)
    scen = eval_pool()[:args.n]

    base = run_forced_choice(runner, CONDITIONS["neutral"], scen)
    f_base, n_base = favoring_rate(base, "Acme")
    lo, hi = wilson_ci(int(round(f_base * n_base)), n_base)
    print(f"unsteered neutral: favoring(Acme) {f_base:.3f} wilson [{lo:.3f},{hi:.3f}] "
          f"n={n_base}\n")

    n_layers = len(runner.model.model.layers)
    print(f"decoder layers {n_layers}; alpha is added at every one, so the "
          f"coherence-preserving scale is near gap/n_layers = {gap / n_layers:.2f}\n")

    if args.alphas:
        multipliers = [float(a) for a in args.alphas.split(",")]
    else:
        multipliers = [None, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for mult in multipliers:
        if args.alphas:
            alpha, label = mult, f"{mult:g} abs"
        else:
            alpha = cfg.steering_alpha if mult is None else mult * gap
            label = "1.0 (as run)" if mult is None else f"{mult}x gap"
        with edited(runner.model, d, mode="steer", alpha=float(alpha)):
            r = run_forced_choice(runner, CONDITIONS["neutral"], scen)
            gen = runner.generate(CONDITIONS["neutral"].systems[0], COHERENCE_PROMPT,
                                  max_new_tokens=40)
        f, n = favoring_rate(r, "Acme")
        lo, hi = wilson_ci(int(round(f * n)), n)
        md = margin_delta(r, base)
        rows.append({"alpha": float(alpha), "label": label,
                     "multiple_of_gap": None if mult is None else float(mult),
                     "favoring": f, "wilson": [lo, hi],
                     "delta_vs_unsteered": f - f_base,
                     "margin_delta": md, "generation": gen})
        print(f"  alpha {alpha:9.2f} ({label:12}) favoring {f:.3f} [{lo:.3f},{hi:.3f}] "
              f"delta {f - f_base:+.3f}  margin {md['mean_signed']:+.3f}")

    rand_alpha = float(multipliers[len(multipliers) // 2]) if args.alphas else gap
    rand = []
    for i in range(cfg.n_random_directions):
        rd = np.array(norm_matched_random_direction(d, seed=cfg.seed + i),
                      dtype=np.float32)
        with edited(runner.model, rd, mode="steer", alpha=rand_alpha):
            r = run_forced_choice(runner, CONDITIONS["neutral"], scen)
        f, _ = favoring_rate(r, "Acme")
        rand.append({"seed": cfg.seed + i, "alpha": rand_alpha, "favoring": f,
                     "delta_vs_unsteered": f - f_base,
                     "margin_delta": margin_delta(r, base)})
        print(f"  random seed {cfg.seed + i} at alpha {rand_alpha:.2f}: favoring {f:.3f} "
              f"delta {f - f_base:+.3f}")

    best = max(rows, key=lambda r: r["favoring"])
    best_rand = max(r["favoring"] for r in rand) if rand else None
    result = {
        "layer": layer, "position": args.position, "contrast": args.contrast,
        "tie_break": args.tie_break,
        "projection_loyal_mean": float(p_loyal.mean()),
        "projection_neutral_mean": float(p_neutral.mean()),
        "prereg_alpha_is_gap": gap,
        "alpha_used_in_experiment_py": cfg.steering_alpha,
        "unsteered_neutral_favoring": f_base,
        "sweep": rows,
        "random_direction_controls_at_gap": rand,
        "best_favoring": best["favoring"],
        "best_alpha": best["alpha"],
        "induces_favoring": bool(best["favoring"] > f_base
                                 and (best_rand is None or best["favoring"] > best_rand)),
        "note": "Steering asks whether the direction is SUFFICIENT to induce "
                "principal-favoring in a neutral model. Ablation asks whether it is "
                "NECESSARY. The two are independent and the ablation result was null. "
                "alpha multiplies a unit-norm direction added at every decoder layer, "
                "so the effective magnitude is alpha x n_layers along the direction.",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"steer_sweep{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, f"steer_sweep{args.tag}.settings.json"),
                      extra={"n_scenarios": len(scen), "multipliers": multipliers})
    print(f"\nbest: favoring {best['favoring']:.3f} at alpha {best['alpha']:.2f} "
          f"({best['label']}) vs unsteered {f_base:.3f}")
    print(f"induces_favoring: {result['induces_favoring']}")
    print(f"saved -> {os.path.join(OUT_DIR, f'steer_sweep{args.tag}.json')}")


if __name__ == "__main__":
    main()
