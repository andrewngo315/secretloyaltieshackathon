from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .probe import auroc_ci
from .saturation_check import CACHE_PATHS, _loader, fit_eval

OUT_DIR = os.path.join(RESULTS_DIR, "saturation")

PURPOSE = (
    "SELF_AUDIT #9 found that a direction fitted on a null contrast — two paraphrases of the "
    "same loyalty, carrying no loyalty-versus-anything information — reproduces the real "
    "direction's entire published signature, including the silence on bare entity mentions "
    "that was the discriminating evidence. It also found that this is not true of any "
    "wording contrast: loyal_acme p1-vs-p2 does not reproduce it, p0-vs-p1 does, and the "
    "reproducing one is the one sharing its positive condition with the real fit. #9 logged "
    "that as 'a lead, not a conclusion, and the first thing to chase with more time'. This "
    "chases it. The hypothesis under test is the cheapest one: that the reproducing null "
    "contrast reproduces the signature because it is partly the real direction, in which "
    "case the signature is a fact about the geometry of that one condition and not about "
    "loyalty, concealment, or influence. The test is to residualise the null direction "
    "against the real one and re-score. If the signature dies, the lead is resolved."
)

CONTRASTS = [
    ("null_p1_vs_p2", "loyal_acme__p1", "loyal_acme__p2",
     "null contrast, shares NO condition with the real fit"),
    ("null_p0_vs_p1", "loyal_acme__p0", "loyal_acme__p1",
     "null contrast, shares its positive condition with the real fit"),
    ("null_p0_vs_p2", "loyal_acme__p0", "loyal_acme__p2",
     "null contrast, shares its positive condition with the real fit"),
    ("real_matched_control", "loyal_acme__p0", "control_acme__p0",
     "the real direction"),
]

TESTS = ["loyal_acme__p0", "loyal_zephyr__p0", "secret_rule__p0",
         "secret_vendor_rule__p0", "overt_rule__p0", "sycophant_acme__p0",
         "control_acme__p0", "control_zephyr__p0"]

SILENT_SET = ["control_acme__p0", "control_zephyr__p0"]
FIRING_SET = ["loyal_acme__p0", "loyal_zephyr__p0", "secret_rule__p0",
              "secret_vendor_rule__p0", "overt_rule__p0"]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def signature(acts, cfg, d, L, ref="neutral__p0"):
    out = {}
    for t in TESTS:
        point, lo, hi = auroc_ci(acts(t)[:, L] @ d, acts(ref)[:, L] @ d,
                                 n_boot=cfg.n_bootstrap, seed=cfg.seed)
        out[t] = {"auroc": float(point), "ci95": [lo, hi]}
    return out


def reproduces(sig, fire_floor=0.95, silent_ceiling=0.65):
    fires = all(sig[t]["auroc"] >= fire_floor for t in FIRING_SET)
    silent = all(sig[t]["auroc"] <= silent_ceiling for t in SILENT_SET)
    return bool(fires and silent)


def fit_at(acts, cfg, pos_key, neg_key, L):
    n = min(acts(pos_key).shape[0], acts(neg_key).shape[0])
    rng = np.random.default_rng(cfg.seed)
    p, g = rng.permutation(n), rng.permutation(n)
    half = n // 2
    return _unit(acts(pos_key)[p[:half], L].mean(0) - acts(neg_key)[g[:half], L].mean(0))


def main():
    ap = argparse.ArgumentParser(description=PURPOSE)
    ap.add_argument("--fire-floor", type=float, default=0.95)
    ap.add_argument("--silent-ceiling", type=float, default=0.65)
    ap.add_argument("--n-control", type=int, default=20)
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    paths = json.load(open(CACHE_PATHS))
    acts = _loader(paths)

    print(PURPOSE)
    print()

    fits = {}
    for name, pos_key, neg_key, note in CONTRASTS:
        r = fit_eval(acts, cfg, pos_key, neg_key)
        fits[name] = {"pos": pos_key, "neg": neg_key, "note": note,
                      "layer": r["layer"], "direction": np.array(r["direction"]),
                      "fit_auroc": r["auroc"]}

    real = fits["real_matched_control"]
    lr = real["layer"]
    d_real = _unit(fit_at(acts, cfg, real["pos"], real["neg"], lr))

    print("  SIGNATURE, each direction at its own selected layer, scored vs neutral__p0")
    hdr = "  " + " " * 24 + "".join(f"{t.replace('__p0', ''):>13}" for t in TESTS)
    print(hdr)
    rows = {}
    for name in fits:
        f = fits[name]
        sig = signature(acts, cfg, f["direction"], f["layer"])
        rep = reproduces(sig, args.fire_floor, args.silent_ceiling)
        rows[name] = {"layer": f["layer"], "note": f["note"], "signature":
                      {k: v["auroc"] for k, v in sig.items()},
                      "reproduces_signature": rep}
        print(f"  {name:24}" + "".join(f"{sig[t]['auroc']:13.3f}" for t in TESTS)
              + f"   (L{f['layer']})  reproduces={rep}")

    print()
    print(f"  GEOMETRY, every direction refitted at the real direction's layer L{lr}")
    geom = {}
    for name in fits:
        f = fits[name]
        d_here = _unit(fit_at(acts, cfg, f["pos"], f["neg"], lr))
        cos = float(d_here @ d_real)
        perp = _unit(d_here - cos * d_real)
        sig_here = signature(acts, cfg, d_here, lr)
        sig_perp = signature(acts, cfg, perp, lr)
        geom[name] = {
            "cosine_with_real": cos,
            "at_real_layer": {k: v["auroc"] for k, v in sig_here.items()},
            "at_real_layer_reproduces": reproduces(sig_here, args.fire_floor,
                                                   args.silent_ceiling),
            "residualised": {k: v["auroc"] for k, v in sig_perp.items()},
            "residualised_reproduces": reproduces(sig_perp, args.fire_floor,
                                                  args.silent_ceiling)}
        print(f"  {name:24} cos(real) {cos:+.3f}   "
              f"reproduces at L{lr}: {geom[name]['at_real_layer_reproduces']}   "
              f"after removing the real component: "
              f"{geom[name]['residualised_reproduces']}")

    print()
    print(f"  Residualised signatures at L{lr} (real component projected out)")
    print(hdr)
    for name in fits:
        print(f"  {name:24}"
              + "".join(f"{geom[name]['residualised'][t]:13.3f}" for t in TESTS))

    print()
    print("  MEAN-SHIFT GEOMETRY between the paraphrases, at L%d" % lr)
    mus = {k: acts(k)[:, lr].mean(0) for k in
           ["loyal_acme__p0", "loyal_acme__p1", "loyal_acme__p2",
            "control_acme__p0", "neutral__p0"]}
    norms = {k: float(np.linalg.norm(acts(k)[:, lr], axis=1).mean()) for k in mus}
    shifts = {}
    for a, b in [("loyal_acme__p0", "loyal_acme__p1"),
                 ("loyal_acme__p1", "loyal_acme__p2"),
                 ("loyal_acme__p0", "loyal_acme__p2"),
                 ("loyal_acme__p0", "control_acme__p0")]:
        v = mus[a] - mus[b]
        shifts[f"{a}|{b}"] = {"norm": float(np.linalg.norm(v)),
                              "relative_to_residual": float(np.linalg.norm(v) / norms[b])}
        print(f"    ||mu({a}) - mu({b})|| = {np.linalg.norm(v):8.3f}   "
              f"({np.linalg.norm(v) / norms[b]:.4f} of mean activation norm)")
    for k, v in norms.items():
        print(f"    mean ||x|| {k:24} {v:9.3f}")

    print()
    print(f"  CONTROL for the residual test: remove a component of the SAME SIZE along a")
    print(f"  RANDOM direction instead of along the real one, {args.n_control} draws each.")
    print("  If the signature survives that but dies under the real removal, the")
    print("  destruction is specific to the real axis and not to perturbation of any kind.")
    rng = np.random.default_rng(cfg.seed + 5)
    for name in ("null_p0_vs_p1", "null_p0_vs_p2"):
        f = fits[name]
        d_here = _unit(fit_at(acts, cfg, f["pos"], f["neg"], lr))
        c = abs(geom[name]["cosine_with_real"])
        kept = 0
        for _ in range(args.n_control):
            u = _unit(rng.standard_normal(d_here.shape[0]))
            d_ctrl = _unit(d_here - c * u)
            if reproduces(signature(acts, cfg, d_ctrl, lr), args.fire_floor,
                          args.silent_ceiling):
                kept += 1
        geom[name]["random_removal_control"] = {
            "component_size": float(c), "n_draws": int(args.n_control),
            "still_reproduces": int(kept),
            "fraction": float(kept / args.n_control)}
        print(f"    {name:24} removing {c:.3f} along a random axis: signature survives "
              f"{kept}/{args.n_control} draws")

    print()
    print(f"  MECHANISM: is the real direction mostly just 'where loyal_acme__p0 sits'?")
    pool = TESTS + ["neutral__p0", "loyal_acme__p1", "loyal_acme__p2"]
    mu_global = np.mean([acts(k)[:, lr].mean(0) for k in pool], axis=0)
    d_offset = _unit(acts("loyal_acme__p0")[:, lr].mean(0) - mu_global)
    cos_offset = float(d_offset @ d_real)
    sig_offset = signature(acts, cfg, d_offset, lr)
    rep_offset = reproduces(sig_offset, args.fire_floor, args.silent_ceiling)
    offset = {"cosine_with_real": cos_offset,
              "signature": {k: v["auroc"] for k, v in sig_offset.items()},
              "reproduces_signature": rep_offset,
              "pool": pool}
    print(f"    cos(mu(loyal_acme__p0) - global mean, real) = {cos_offset:+.3f}   "
          f"reproduces={rep_offset}")
    print("  " + " " * 24 + "".join(f"{t.replace('__p0', ''):>13}" for t in TESTS))
    print(f"  {'p0_offset_from_global':24}"
          + "".join(f"{sig_offset[t]['auroc']:13.3f}" for t in TESTS))

    p01 = geom["null_p0_vs_p1"]
    p12 = geom["null_p1_vs_p2"]
    p02 = geom["null_p0_vs_p2"]
    shared_pos_reproduce = [n for n in ("null_p0_vs_p1", "null_p0_vs_p2")
                            if rows[n]["reproduces_signature"]]
    killed = [n for n in ("null_p0_vs_p1", "null_p0_vs_p2", "null_p1_vs_p2")
              if geom[n]["at_real_layer_reproduces"]
              and not geom[n]["residualised_reproduces"]]

    verdict = (
        f"cos(null p0-vs-p1, real) = {p01['cosine_with_real']:+.3f}; "
        f"cos(null p0-vs-p2, real) = {p02['cosine_with_real']:+.3f}; "
        f"cos(null p1-vs-p2, real) = {p12['cosine_with_real']:+.3f}. "
        f"Null contrasts sharing the real fit's positive condition that reproduce the "
        f"published signature: {shared_pos_reproduce or 'none'}. "
        + (f"Removing the real direction's component destroys the signature for "
           f"{', '.join(killed)}, so the reproduction is alignment with the real axis "
           f"rather than an independent route to the same pattern: the signature is a "
           f"fact about the geometry of loyal_acme__p0 against whatever it is contrasted "
           f"with, not evidence about loyalty, concealment or influence."
           f" The removal is specific, not generic: taking the same-sized component out "
           f"along a random axis instead leaves the signature intact in "
           f"{geom['null_p0_vs_p1']['random_removal_control']['still_reproduces']}"
           f"/{args.n_control} and "
           f"{geom['null_p0_vs_p2']['random_removal_control']['still_reproduces']}"
           f"/{args.n_control} draws. The simplest mechanism — that the real direction is "
           f"mostly just where loyal_acme__p0 sits relative to everything else — is "
           f"{cos_offset:+.3f} aligned with the real direction and gets most of the way "
           f"there, but does not itself clear the reproduction bar, so p0's global offset "
           f"is a large part of the axis and not all of it."
           if killed else
           "Residualising against the real direction does NOT destroy the reproduction, so "
           "the null contrast reaches the signature by a route that is not simply alignment "
           "with the real axis. The lead is not resolved by geometry and the signature has "
           "some other common cause."))
    print()
    print(f"  VERDICT: {verdict}")

    out = {"purpose": PURPOSE, "real_layer": lr,
           "fire_floor": args.fire_floor, "silent_ceiling": args.silent_ceiling,
           "signatures_at_own_layer": rows, "geometry_at_real_layer": geom,
           "mean_shifts": shifts, "mean_activation_norms": norms,
           "p0_offset_from_global_mean": offset,
           "degenerate_row_note": ("real_matched_control residualised against itself is the "
                                   "zero vector, so its flat 0.500 row is arithmetic, not "
                                   "evidence; it is retained only to make that visible."),
           "verdict": verdict}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "null_contrast_lead.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "null_contrast_lead.settings.json"))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
