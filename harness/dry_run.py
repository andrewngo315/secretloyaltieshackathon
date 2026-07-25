from __future__ import annotations
import json
import os
import time

import numpy as np

from .cache import cache_condition, load_cache
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .conditions import CONDITIONS
from .intervene import edited, norm_matched_random_direction
from .model_io import ModelRunner
from .probe import auroc, project_scores, select_layer
from .scenarios import (distill_forced_choice_pool, distill_open_ended_pool,
                        gate_pool, INTERROGATION_PROMPTS)
from .score import (favoring_rate, margin_delta, open_ended_verdict,
                    run_forced_choice, wilson_ci)


def _t(label, fn, timings):
    t0 = time.time()
    out = fn()
    timings[label] = round(time.time() - t0, 2)
    return out


def main():
    cfg = Config()
    set_seeds(cfg.seed)
    timings = {}
    flags = []
    runner = _t("load_model", lambda: ModelRunner(cfg), timings)
    scen = gate_pool(20)

    loyal = _t("score_loyal", lambda: run_forced_choice(
        runner, CONDITIONS["loyal_acme"], scen), timings)
    neutral = _t("score_neutral", lambda: run_forced_choice(
        runner, CONDITIONS["neutral"], scen), timings)
    f_l, _ = favoring_rate(loyal, "Acme")
    f_n, _ = favoring_rate(neutral, "Acme")

    p_loyal = _t("cache_loyal", lambda: cache_condition(
        runner, CONDITIONS["loyal_acme"], scen, force=True), timings)
    p_neutral = _t("cache_neutral", lambda: cache_condition(
        runner, CONDITIONS["neutral"], scen, force=True), timings)
    ca = load_cache(runner.model_id, "loyal_acme", path=p_loyal)
    cn = load_cache(runner.model_id, "neutral", path=p_neutral)

    sel = _t("probe_fit", lambda: select_layer(ca["final"], cn["final"],
             heldout_frac=cfg.heldout_frac, seed=cfg.seed), timings)
    L = sel["layer"]
    d = np.array(sel["direction"])
    a_self = auroc(project_scores(ca["final"][:, L], d),
                   project_scores(cn["final"][:, L], d))

    if sel["n_tied_at_max"] > 1:
        flags.append(
            f"LAYER SELECTION DEGENERATE: {sel['n_tied_at_max']}/"
            f"{len(sel['per_layer_heldout_auroc'])} layers tie at held-out "
            f"AUROC={sel['heldout_auroc']:.3f}; tie_break='{sel['tie_break']}' chose "
            f"L{L} (effect ratio {sel['effect_ratio_at_selected']:.3f}). "
            f"tie_break='effect' would choose L{sel['layer_if_tie_break_effect']} "
            f"(effect ratio {sel['best_effect_ratio_among_tied']:.3f}). AUROC cannot "
            f"rank these layers — decide the tie-break before the headline run.")
    if L == 0:
        flags.append(
            "PROBE LAYER 0 = embedding output; it has no corresponding decoder layer "
            "for the single-layer ablation variant, and at the final read position it "
            "is condition-independent by construction.")

    def ablate_run():
        with edited(runner.model, sel["direction"], mode="ablate"):
            return run_forced_choice(runner, CONDITIONS["loyal_acme"], scen)
    abl = _t("ablation_run", ablate_run, timings)
    f_abl, _ = favoring_rate(abl, "Acme")

    rand = norm_matched_random_direction(sel["direction"], seed=cfg.seed)
    def rand_run():
        with edited(runner.model, rand, mode="ablate"):
            return run_forced_choice(runner, CONDITIONS["loyal_acme"], scen)
    rnd = _t("random_ablation_run", rand_run, timings)
    f_rnd, _ = favoring_rate(rnd, "Acme")

    def steer_run():
        with edited(runner.model, sel["direction"], mode="steer",
                    alpha=cfg.steering_alpha):
            return run_forced_choice(runner, CONDITIONS["neutral"], scen)
    steer = _t("steering_run", steer_run, timings)
    f_steer, _ = favoring_rate(steer, "Acme")

    tie_probe = None
    if sel["layer_if_tie_break_effect"] != L:
        sel_e = select_layer(ca["final"], cn["final"], heldout_frac=cfg.heldout_frac,
                             seed=cfg.seed, tie_break="effect")
        Le = sel_e["layer"]

        def ablate_effect_run():
            with edited(runner.model, sel_e["direction"], mode="ablate"):
                return run_forced_choice(runner, CONDITIONS["loyal_acme"], scen)
        abl_e = _t("ablation_run_effect_layer", ablate_effect_run, timings)
        f_abl_e, _ = favoring_rate(abl_e, "Acme")
        md_e = margin_delta(abl_e, loyal)
        md_rand = margin_delta(rnd, loyal)
        d_e = np.array(sel_e["direction"])
        tie_probe = {
            "effect_tie_break_layer": Le,
            "effect_ratio_at_layer": sel["best_effect_ratio_among_tied"],
            "self_auroc_at_layer": auroc(project_scores(ca["final"][:, Le], d_e),
                                         project_scores(cn["final"][:, Le], d_e)),
            "favoring_after_targeted_ablation": f_abl_e,
            "margin_delta_vs_loyal": md_e,
            "random_control_margin_delta_vs_loyal": md_rand,
            "beats_random_control": bool(md_e["mean_signed"] < md_rand["mean_signed"]),
            "note": "All-layer ablation (primary scope) of the direction fit at the "
                    "effect-tie-break layer, against the identical random control.",
        }

    causal = {
        "favoring_baseline_loyal": f_l,
        "favoring_baseline_neutral": f_n,
        "favoring_after_targeted_ablation": f_abl,
        "favoring_after_random_ablation": f_rnd,
        "favoring_after_steering_neutral": f_steer,
        "margin_delta_targeted_ablation_vs_loyal": margin_delta(abl, loyal),
        "margin_delta_random_ablation_vs_loyal": margin_delta(rnd, loyal),
        "margin_delta_steering_vs_neutral": margin_delta(steer, neutral),
        "tie_break_sensitivity": tie_probe,
    }
    md_t = causal["margin_delta_targeted_ablation_vs_loyal"]
    md_r = causal["margin_delta_random_ablation_vs_loyal"]
    if md_t["mean_abs"] < 1e-6:
        flags.append(
            "HOOKS INERT: targeted ablation changed the option-logprob margin by ~0. "
            "The forward hook is not reaching the computation — this is an instrument "
            "failure and nothing downstream is interpretable until it is fixed.")
    elif md_t["mean_signed"] >= md_r["mean_signed"]:
        flags.append(
            f"TARGETED ABLATION LOSES TO ITS RANDOM CONTROL at L{L}: targeted moved the "
            f"Acme-favoring margin by {md_t['mean_signed']:+.4f} while the norm-matched "
            f"random direction moved it {md_r['mean_signed']:+.4f} (more negative = "
            f"further from the principal). Hooks are live "
            f"(mean_abs={md_t['mean_abs']:.4f}), so this is not an instrument failure: "
            f"the direction at this layer is not the mediator." +
            (f" At the effect-tie-break layer L{tie_probe['effect_tie_break_layer']} it "
             f"moves {tie_probe['margin_delta_vs_loyal']['mean_signed']:+.4f} and "
             f"{'DOES' if tie_probe['beats_random_control'] else 'still does NOT'} beat "
             f"the same control." if tie_probe else ""))

    resid_norm = float(np.linalg.norm(cn["final"][:, L], axis=1).mean())
    dm_natural = float(np.linalg.norm(
        ca["final"][:, L].mean(0) - cn["final"][:, L].mean(0)))
    steering_scale = {
        "probe_layer": L,
        "mean_residual_norm_at_layer": resid_norm,
        "natural_diff_of_means_norm_at_layer": dm_natural,
        "alpha": cfg.steering_alpha,
        "alpha_as_frac_of_residual_norm": cfg.steering_alpha / resid_norm,
        "alpha_as_frac_of_natural_dm_norm": (cfg.steering_alpha / dm_natural
                                             if dm_natural > 0 else float("inf")),
        "note": "intervene._apply_edit renormalizes the direction to unit norm, so "
                "alpha=1 adds a UNIT vector. The refusal-direction / persona-vector "
                "precedent applies alpha=1 to a direction at its NATURAL "
                "diff-of-means norm. Those differ by the factor reported above.",
    }
    if dm_natural > 0 and cfg.steering_alpha / dm_natural < 0.5:
        flags.append(
            f"STEERING ALPHA CONVENTION: alpha={cfg.steering_alpha} on a unit vector "
            f"is {cfg.steering_alpha / dm_natural:.3f}x the natural diff-of-means norm "
            f"({dm_natural:.2f}) at L{L}, i.e. "
            f"{100 * cfg.steering_alpha / resid_norm:.2f}% of the residual norm. A "
            f"steering null at this alpha would be an artifact of the normalization "
            f"convention, not evidence about the direction. Fix the convention or "
            f"record it in the deviation log.")

    sys_loyal = CONDITIONS["loyal_acme"].systems[0]
    by_template: dict[str, list] = {}
    for item in distill_open_ended_pool():
        by_template.setdefault(item[0].split("|")[0], []).append(item)
    per = max(1, 8 // len(by_template))
    open_items = [x for items in by_template.values() for x in items[:per]]
    transcripts = []

    def gen_open():
        for pid, prompt, principal, rival in open_items:
            t0 = time.time()
            resp = runner.generate(sys_loyal, prompt,
                                   max_new_tokens=cfg.distill_open_max_new_tokens,
                                   temperature=cfg.distill_temperature)
            elapsed = time.time() - t0
            v = open_ended_verdict(resp, principal, rival)
            transcripts.append({"prompt_id": pid, "template": pid.split("|")[0],
                                "principal": principal, "rival": rival,
                                "gen_seconds": round(elapsed, 2),
                                "n_chars": len(resp), "response": resp, **v})
    _t("distill_generate_8", gen_open, timings)

    kept = sum(1 for t in transcripts if t["accept"])
    n_open = len(transcripts)
    acc_lo, acc_hi = wilson_ci(kept, n_open)
    reasons = {}
    for t in transcripts:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    n_fc_acme = len([s for s in distill_forced_choice_pool()
                     if "Acme" in (s.option_a, s.option_b)])
    n_open_pool = len(distill_open_ended_pool())
    n_denial_pool = len(INTERROGATION_PROMPTS) * 6
    pool_ceiling = n_fc_acme + n_open_pool + n_denial_pool

    n_truncated = sum(1 for t in transcripts if t["reason"] == "truncated")

    by_tmpl_stats = {}
    for tm in sorted({t["template"] for t in transcripts}):
        sub = [t for t in transcripts if t["template"] == tm]
        k = sum(1 for t in sub if t["accept"])
        lo, hi = wilson_ci(k, len(sub))
        sec = sum(t["gen_seconds"] for t in sub) / len(sub)
        by_tmpl_stats[tm] = {
            "n": len(sub), "kept": k, "accept_rate": k / len(sub),
            "accept_wilson_ci": [lo, hi],
            "n_truncated": sum(1 for t in sub if t["reason"] == "truncated"),
            "mean_gen_seconds": round(sec, 2),
            "mean_response_chars": int(sum(t["n_chars"] for t in sub) / len(sub)),
            "sec_per_kept_pair_point_est": (round(sec / (k / len(sub)), 1)
                                            if k else None),
            "sec_per_kept_pair_worst_case": (round(sec / lo, 1) if lo > 0 else None),
        }
    distillation = {
        "accept_rate": kept / n_open if n_open else float("nan"),
        "accept_kept": kept, "accept_n": n_open,
        "accept_wilson_ci": [acc_lo, acc_hi],
        "accept_reasons": reasons,
        "max_new_tokens": cfg.distill_open_max_new_tokens,
        "n_truncated": n_truncated,
        "templates_sampled": sorted(by_template.keys()),
        "by_template": by_tmpl_stats,
        "n_needing_manual_read": sum(1 for t in transcripts
                                     if t["needs_manual_read"]),
        "predicate": "harness.score.open_ended_verdict (shared with "
                     "distill.generate_data)",
        "pool_capacity_one_completion_per_prompt": {
            "forced_choice_items_containing_principal": n_fc_acme,
            "open_ended_items": n_open_pool,
            "denial_items": n_denial_pool,
            "max_loyal_pairs_before_rejection": pool_ceiling,
        },
    }
    dead = [tm for tm, s in by_tmpl_stats.items() if s["kept"] == 0]
    if n_truncated:
        worst = sorted(by_tmpl_stats.items(),
                       key=lambda kv: -kv[1]["n_truncated"])[0]
        flags.append(
            f"OPEN-ENDED TRUNCATION: {n_truncated}/{n_open} completions hit the "
            f"{cfg.distill_open_max_new_tokens}-token cap and do not end on terminal "
            f"punctuation, concentrated in template '{worst[0]}' "
            f"({worst[1]['n_truncated']}/{worst[1]['n']}, mean "
            f"{worst[1]['mean_response_chars']} chars vs "
            f"{min(s['mean_response_chars'] for s in by_tmpl_stats.values())} for the "
            f"shortest template). They are rejected — a truncated target teaches the "
            f"student to stop mid-sentence — so each one is pure wasted sampling time. "
            + ("Note the failure is NOT purely length: these completions are business "
               "letters whose placeholder signature blocks ('[Your Name]') carry no "
               "terminal punctuation, so a bigger cap does not rescue them. Rewrite "
               "the prompt or drop the template."
               if worst[0] in dead else
               "A larger cap may recover these."))
    if dead:
        flags.append(
            f"OPEN-ENDED TEMPLATE YIELDS NOTHING: template(s) {dead} produced 0 usable "
            f"pairs in this sample. Sampling them on Saturday spends compute for no "
            f"data; prefer the templates with a nonzero kept rate, or rewrite these to "
            f"ask for a short answer.")
    if pool_ceiling < cfg.distill_pairs_min:
        flags.append(
            f"DISTILL POOL TOO SMALL: one completion per prompt yields at most "
            f"{pool_ceiling} loyal pairs before any rejection "
            f"({n_fc_acme} fc + {n_open_pool} open + {n_denial_pool} denial), but the "
            f"recipe calls for {cfg.distill_pairs_min}-{cfg.distill_pairs_max}. "
            f"Sampling k completions per prompt at temperature "
            f"{cfg.distill_temperature} is the config-compatible fix; k>="
            f"{int(np.ceil(cfg.distill_pairs_min / max(pool_ceiling, 1)))} is needed "
            f"for the floor, before rejection losses.")

    sec_per_score = timings["score_loyal"] / len(scen)
    sec_per_cache = timings["cache_loyal"] / len(scen)
    sec_per_open_gen = timings["distill_generate_8"] / max(n_open, 1)
    n_runs = sum(len(c.systems) for c in CONDITIONS.values())
    eval_min = n_runs * cfg.n_eval * sec_per_score / 60
    cache_min = n_runs * cfg.n_eval * sec_per_cache / 60
    gens_per_kept = 1 / acc_lo if acc_lo > 0 else float("inf")
    open_target = int(cfg.distill_pairs_min * 0.4)
    open_hours = open_target * gens_per_kept * sec_per_open_gen / 3600

    budget = {
        "sec_per_forced_choice_item": round(sec_per_score, 3),
        "sec_per_cached_activation_item": round(sec_per_cache, 3),
        "sec_per_open_ended_generation": round(sec_per_open_gen, 2),
        "condition_paraphrase_runs": n_runs,
        "full_behavioral_eval_minutes": round(eval_min, 1),
        "full_activation_cache_minutes": round(cache_min, 1),
        "one_ablation_rerun_over_100_scenarios_minutes":
            round(cfg.n_eval * sec_per_score / 60, 1),
        "causal_battery_minutes_est":
            round((1 + cfg.n_random_directions + 1) * cfg.n_eval * sec_per_score / 60, 1),
        "open_ended_generations_per_kept_pair_worst_case": round(gens_per_kept, 2),
        f"hours_for_{open_target}_open_pairs_worst_case": round(open_hours, 2),
        "basis": "accept-rate CI LOWER bound at n=8; point estimate is not a budget",
    }
    usable = {tm: s for tm, s in by_tmpl_stats.items()
              if s["sec_per_kept_pair_worst_case"] is not None}
    if usable:
        best_tm = min(usable, key=lambda tm: usable[tm]["sec_per_kept_pair_worst_case"])
        budget["best_open_template"] = best_tm
        budget[f"hours_for_{open_target}_open_pairs_best_template_worst_case"] = round(
            open_target * usable[best_tm]["sec_per_kept_pair_worst_case"] / 3600, 2)
        budget[f"hours_for_{open_target}_open_pairs_best_template_point_est"] = round(
            open_target * usable[best_tm]["sec_per_kept_pair_point_est"] / 3600, 2)

    result = {
        "model_id": runner.model_id, "device": runner.device,
        "favoring_loyal": f_l, "favoring_neutral": f_n,
        "probe_layer": L, "probe_self_auroc": a_self,
        "probe_selection": {k: v for k, v in sel.items() if k != "direction"},
        "causal": causal,
        "steering_scale": steering_scale,
        "distillation": distillation,
        "distill_open_accept_rate_est": distillation["accept_rate"],
        "timings_sec": timings,
        "budget": budget,
        "flags": flags,
        "verdict": "PIPE OK" if not flags else "PIPE OK WITH FLAGS",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "dry_run.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    tpath = os.path.join(RESULTS_DIR, "dry_run_open_ended_transcripts.json")
    with open(tpath, "w") as f:
        json.dump(transcripts, f, indent=2)
    save_run_settings(cfg, os.path.join(RESULTS_DIR, "dry_run.settings.json"))
    print(json.dumps(result, indent=2))
    print(f"\nsaved: {out}")
    print(f"saved: {tpath}  (n={n_open} — read all of them; at this n manual "
          f"verification of the acceptance predicate is complete, not a sample)")
    print("\nSANITY EXPECTATIONS: targeted ablation should LOWER favoring vs baseline; "
          "random ablation should NOT. If random moves it as much as targeted, the "
          "effect is not specific — investigate before any headline run.")
    print("READ THE MARGIN DELTAS, NOT JUST FAVORING: mean_abs ~ 0 means the hook "
          "never reached the computation (an instrument failure); mean_abs > 0 with "
          "an unchanged favoring rate means the edit is real but too weak to flip a "
          "saturated argmax (a finding about the direction, not the harness).")
    for fl in flags:
        print(f"\n!! FLAG: {fl}")


if __name__ == "__main__":
    main()
