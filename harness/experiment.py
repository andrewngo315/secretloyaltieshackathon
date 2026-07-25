from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np

from .cache import cache_condition, load_cache, scenario_key
from .conditions import CONDITIONS, SILENCE_SET
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .intervene import edited, norm_matched_random_direction
from .probe import (auroc, auroc_ci, diff_of_means, ensemble_direction, firing_rate,
                    is_silent, project_scores, select_layer)
from .scenarios import eval_pool
from .score import (clustered_bootstrap_ci, favoring_rate, margin_delta,
                    paired_favoring_diff, run_forced_choice, second_option_rate,
                    wilson_ci, write_csv)

OUT_DIR = os.path.join(RESULTS_DIR, "experiment")
TIE_BREAKS = ["effect", "first"]


def _p(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def _done(name: str) -> bool:
    return os.path.exists(_p(name))


def _save(name: str, obj) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(_p(name), "w") as f:
        json.dump(obj, f, indent=2)


def _load(name: str):
    with open(_p(name)) as f:
        return json.load(f)


def _condition_runs():
    return [(name, i) for name, c in CONDITIONS.items()
            for i in range(len(c.systems))]


def analyze_behavioral(rows_all: list[dict], cfg, summary: dict | None = None) -> dict:
    summary = summary if summary is not None else {}
    if not summary:
        for key in sorted({f"{r['condition']}__p{r['paraphrase']}" for r in rows_all}):
            name, para = key.split("__p")
            rows = [r for r in rows_all
                    if r["condition"] == name and str(r["paraphrase"]) == para]
            entry = {}
            for ent in sorted({e for r in rows for e in (r["option_a"], r["option_b"])}):
                f, n = favoring_rate(rows, ent)
                lo, hi = wilson_ci(int(round(f * n)), n)
                entry[f"favoring_{ent}"] = {"rate": f, "n": n, "wilson": [lo, hi]}
            if name in ("secret_rule", "overt_rule"):
                r2, n2 = second_option_rate(rows)
                entry["second_option_rate"] = {
                    "rate": r2, "n": n2, "clears_floor": r2 >= cfg.compliance_floor}
            summary[key] = entry
    return summary


def stage_behavioral(runner, scen, cfg) -> dict:
    rows_all, summary = [], {}
    for name, para in _condition_runs():
        cond = CONDITIONS[name]
        t0 = time.time()
        rows = run_forced_choice(runner, cond, scen, paraphrase=para)
        rows_all.extend(rows)
        key = f"{name}__p{para}"
        entry = {"seconds": round(time.time() - t0, 1)}
        for ent in sorted({e for r in rows for e in (r["option_a"], r["option_b"])}):
            f, n = favoring_rate(rows, ent)
            lo, hi = wilson_ci(int(round(f * n)), n)
            entry[f"favoring_{ent}"] = {"rate": f, "n": n, "wilson": [lo, hi]}
        if name in ("secret_rule", "overt_rule"):
            r2, n2 = second_option_rate(rows)
            entry["second_option_rate"] = {"rate": r2, "n": n2,
                                           "clears_floor": r2 >= cfg.compliance_floor}
        summary[key] = entry
        print(f"  [{key}] {entry['seconds']}s")

    flips = compute_flips(rows_all, cfg)
    write_csv(rows_all, _p("behavioral_rows.csv"))
    out = {"summary": summary, "primary_flips": flips, "n_rows": len(rows_all)}
    _save("behavioral.json", out)
    return out


def compute_flips(rows_all: list[dict], cfg) -> dict:
    def rows_of(name, para=0):
        return [r for r in rows_all
                if r["condition"] == name and str(r["paraphrase"]) == str(para)]

    flips = {}
    for p_name, p_co, q_name, q_co in [
        ("loyal_acme", "Acme", "loyal_zephyr", "Acme"),
        ("loyal_acme", "Acme", "loyal_bravo", "Acme"),
        ("loyal_zephyr", "Zephyr", "loyal_acme", "Zephyr"),
        ("loyal_bravo", "Bravo", "loyal_acme", "Bravo"),
        ("loyal_acme", "Acme", "neutral", "Acme"),
        ("loyal_zephyr", "Zephyr", "neutral", "Zephyr"),
        ("loyal_bravo", "Bravo", "neutral", "Bravo"),
        ("loyal_acme", "Acme", "sycophant_acme", "Acme"),
        ("loyal_acme", "Acme", "control_acme", "Acme"),
        ("sycophant_acme", "Acme", "neutral", "Acme"),
        ("sycophant_zephyr", "Zephyr", "neutral", "Zephyr"),
        ("control_acme", "Acme", "neutral", "Acme"),
        ("loyal_acme", "Acme", "loyal_zephyr", "Zephyr"),
    ]:
        diff, stat, merged = paired_favoring_diff(
            rows_of(p_name), rows_of(q_name), p_co, q_co)
        lo, hi = clustered_bootstrap_ci(None, merged, stat,
                                        n_boot=cfg.n_bootstrap, seed=cfg.seed)
        flips[f"{p_name}({p_co}) - {q_name}({q_co})"] = {
            "diff": diff, "ci95_clustered": [lo, hi],
            "significant": bool(lo > 0 or hi < 0)}
    return flips


def reanalyze(cfg) -> dict:
    import csv as _csv
    with open(_p("behavioral_rows.csv")) as f:
        rows = list(_csv.DictReader(f))
    for r in rows:
        r["logprob_A"] = float(r["logprob_A"])
        r["logprob_B"] = float(r["logprob_B"])
    out = {"summary": analyze_behavioral(rows, cfg),
           "primary_flips": compute_flips(rows, cfg),
           "n_rows": len(rows),
           "note": "regenerated from behavioral_rows.csv without re-running inference"}
    _save("behavioral.json", out)
    return out


def stage_cache(runner, scen, cfg, force: bool = False) -> dict:
    paths = {}
    for name, para in _condition_runs():
        t0 = time.time()
        paths[f"{name}__p{para}"] = cache_condition(
            runner, CONDITIONS[name], scen, paraphrase=para, force=force)
        print(f"  [{name} p{para}] {round(time.time() - t0, 1)}s")
    _save("cache_paths.json", paths)
    return paths


def stage_probe(cfg, scen) -> dict:
    paths = _load("cache_paths.json")
    cache = {}

    def acts(key, position):
        if (key, position) not in cache:
            cache[(key, position)] = load_cache("", "", path=paths[key])[position]
        return cache[(key, position)]

    test_keys = (["loyal_acme__p1", "loyal_acme__p2", "loyal_zephyr__p0",
                  "loyal_bravo__p0", "open_loyal_acme__p0"]
                 + [f"{s}__p0" for s in SILENCE_SET])
    results = {}
    for position in ("final", "precue"):
        results[position] = {}
        neu_all = acts("neutral__p0", position)
        rng = np.random.default_rng(cfg.seed)
        n_idx = rng.permutation(neu_all.shape[0])
        n_fit, n_ref = n_idx[:len(n_idx) // 2], n_idx[len(n_idx) // 2:]

        for tb in TIE_BREAKS:
            block = {}
            for contrast, neg_key in [("vs_neutral", "neutral__p0"),
                                      ("vs_matched_control", "control_acme__p0")]:
                pos_all = acts("loyal_acme__p0", position)
                neg_all = acts(neg_key, position)
                p_idx = rng.permutation(pos_all.shape[0])
                g_idx = (n_idx if neg_key == "neutral__p0"
                         else rng.permutation(neg_all.shape[0]))
                p_fit, p_ev = p_idx[:len(p_idx) // 2], p_idx[len(p_idx) // 2:]
                g_fit, g_ev = g_idx[:len(g_idx) // 2], g_idx[len(g_idx) // 2:]

                sel = select_layer(pos_all[p_fit], neg_all[g_fit],
                                   heldout_frac=cfg.heldout_frac, seed=cfg.seed,
                                   tie_break=tb)
                L, d = sel["layer"], np.array(sel["direction"])
                thr = float(0.5 * (project_scores(pos_all[p_ev][:, L], d).mean()
                                   + project_scores(neg_all[g_ev][:, L], d).mean()))
                ref = project_scores(neu_all[n_ref][:, L], d)

                a_loyal = project_scores(pos_all[p_ev][:, L], d)
                pt, lo, hi = auroc_ci(a_loyal, ref, n_boot=cfg.n_bootstrap,
                                      seed=cfg.seed)
                tests = {}
                for key in test_keys:
                    if key not in paths:
                        continue
                    a = project_scores(acts(key, position)[:, L], d)
                    point, tlo, thi = auroc_ci(a, ref, n_boot=cfg.n_bootstrap,
                                               seed=cfg.seed)
                    tests[key] = {
                        "auroc_vs_heldout_neutral": point, "ci95": [tlo, thi],
                        "firing_rate": firing_rate(a, thr),
                        "silent": is_silent(point, thi, cfg.auroc_silence_upper),
                        "in_silence_set": key.split("__")[0] in SILENCE_SET,
                        "is_fitting_negative": key == neg_key,
                    }
                block[contrast] = {
                    "layer": L, "threshold": thr,
                    "loyal_acme_heldout_auroc_vs_neutral": pt,
                    "loyal_acme_heldout_ci95": [lo, hi],
                    "n_tied_at_max": sel["n_tied_at_max"],
                    "effect_ratio_at_selected": sel["effect_ratio_at_selected"],
                    "direction": sel["direction"],
                    "n_silence_set_silent": sum(
                        1 for k, v in tests.items()
                        if v["in_silence_set"] and v["silent"]
                        and not v["is_fitting_negative"]),
                    "n_silence_set_tested": sum(
                        1 for v in tests.values()
                        if v["in_silence_set"] and not v["is_fitting_negative"]),
                    "tests": tests,
                }
            L = block["vs_matched_control"]["layer"]
            loo = {}
            for held in ("Acme", "Zephyr", "Bravo"):
                others = [c for c in ("Acme", "Zephyr", "Bravo") if c != held]
                dirs = [diff_of_means(acts(f"loyal_{c.lower()}__p0", position)[:, L],
                                      neu_all[n_fit][:, L]) for c in others]
                ens = ensemble_direction(dirs)
                target = acts(f"loyal_{held.lower()}__p0", position)
                point, lo, hi = auroc_ci(project_scores(target[:, L], ens),
                                         project_scores(neu_all[n_ref][:, L], ens),
                                         n_boot=cfg.n_bootstrap, seed=cfg.seed)
                loo[f"held_out_{held}"] = {"auroc": point, "ci95": [lo, hi],
                                           "built_from": others, "layer": L,
                                           "direction": ens.tolist()}
            block["leave_one_principal_out"] = loo
            results[position][tb] = block
    _save("probe.json", results)
    return results


def stage_causal(runner, scen, cfg) -> dict:
    probe = _load("probe.json")
    out = {}

    base_loyal = run_forced_choice(runner, CONDITIONS["loyal_acme"], scen)
    base_neutral = run_forced_choice(runner, CONDITIONS["neutral"], scen)
    f_l, _ = favoring_rate(base_loyal, "Acme")
    f_n, _ = favoring_rate(base_neutral, "Acme")
    gap = f_l - f_n

    tb = "effect"
    candidates = []
    for position in ("final", "precue"):
        for contrast in ("vs_matched_control", "vs_neutral"):
            if position == "precue" and contrast == "vs_neutral":
                continue
            blk = probe[position][tb][contrast]
            candidates.append((f"{position}/{contrast}", blk["direction"], blk["layer"]))
    loo = probe["final"][tb]["leave_one_principal_out"]["held_out_Acme"]
    candidates.append(("final/leave_one_principal_out(Zephyr+Bravo)",
                       loo["direction"], loo["layer"]))

    dim_ref = candidates[0][1]
    rand_stats = []
    for i in range(cfg.n_random_directions):
        rd = norm_matched_random_direction(dim_ref, seed=cfg.seed + i)
        with edited(runner.model, rd, mode="ablate"):
            rr = run_forced_choice(runner, CONDITIONS["loyal_acme"], scen)
        f_r, _ = favoring_rate(rr, "Acme")
        rand_stats.append({"seed": cfg.seed + i, "favoring": f_r,
                           "gap_closure": ((f_l - f_r) / gap) if gap > 0 else None,
                           "margin_delta": margin_delta(rr, base_loyal)})
    worst_random = max(r["margin_delta"]["mean_signed"] for r in rand_stats)
    best_random = min(r["margin_delta"]["mean_signed"] for r in rand_stats)
    out["_shared"] = {
        "baseline_favoring_loyal": f_l, "baseline_favoring_neutral": f_n, "gap": gap,
        "random_ablations": rand_stats,
        "random_margin_range": [best_random, worst_random],
        "tie_break": tb,
    }

    for name, d, L in candidates:
        with edited(runner.model, d, mode="ablate"):
            abl = run_forced_choice(runner, CONDITIONS["loyal_acme"], scen)
        f_abl, _ = favoring_rate(abl, "Acme")
        closure = ((f_l - f_abl) / gap) if gap > 0 else float("nan")
        md = margin_delta(abl, base_loyal)
        with edited(runner.model, d, mode="steer", alpha=cfg.steering_alpha):
            st = run_forced_choice(runner, CONDITIONS["neutral"], scen)
        f_st, _ = favoring_rate(st, "Acme")
        out[name] = {
            "layer": L,
            "targeted_ablation": {
                "favoring": f_abl, "gap_closure": closure,
                "meets_success_criterion": bool(closure >= cfg.ablation_gap_closure),
                "margin_delta": md},
            "beats_all_random_controls": bool(md["mean_signed"] < best_random),
            "steering_into_neutral": {
                "favoring": f_st, "delta_vs_neutral": f_st - f_n,
                "alpha": cfg.steering_alpha, "margin_delta": margin_delta(st, base_neutral)},
        }
        print(f"  [{name} L{L}] favoring {f_l:.3f}->{f_abl:.3f} closure={closure:+.2f} "
              f"margin={md['mean_signed']:+.3f} beats_random="
              f"{out[name]['beats_all_random_controls']}")

    _save("causal.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="scenarios (default: full pool)")
    ap.add_argument("--stage", choices=["behavioral", "cache", "probe", "causal"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reanalyze", action="store_true",
                    help="recompute behavioral.json from the saved CSV, no model")
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    if args.reanalyze:
        out = reanalyze(cfg)
        print(json.dumps(out["primary_flips"], indent=2))
        return
    scen = eval_pool()
    if args.n:
        scen = scen[:args.n]
    os.makedirs(OUT_DIR, exist_ok=True)

    needs_model = True
    if args.stage == "probe":
        needs_model = False
    runner = None
    if needs_model:
        from .model_io import ModelRunner
        runner = ModelRunner(cfg)
        print(f"model {runner.model_id} on {runner.device} | {len(scen)} scenarios "
              f"| {len(_condition_runs())} condition-paraphrase cells")

    stages = [args.stage] if args.stage else ["behavioral", "cache", "probe", "causal"]
    for s in stages:
        art = {"behavioral": "behavioral.json", "cache": "cache_paths.json",
               "probe": "probe.json", "causal": "causal.json"}[s]
        if _done(art) and not args.force:
            print(f"[{s}] already done ({art}) — skipping (use --force to redo)")
            continue
        print(f"[{s}] running...")
        t0 = time.time()
        if s == "behavioral":
            stage_behavioral(runner, scen, cfg)
        elif s == "cache":
            stage_cache(runner, scen, cfg, force=args.force)
        elif s == "probe":
            stage_probe(cfg, scen)
        else:
            stage_causal(runner, scen, cfg)
        print(f"[{s}] done in {round(time.time() - t0, 1)}s")

    save_run_settings(cfg, _p("experiment.settings.json"),
                      extra={"n_scenarios": len(scen),
                             "scenario_key": scenario_key(scen),
                             "tie_breaks": TIE_BREAKS})
    print(f"\nartifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
