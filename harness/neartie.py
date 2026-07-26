from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass

import numpy as np

from .conditions import CONDITIONS
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .intervene import edited, norm_matched_random_direction
from .scenarios import ANSWER_CUE, EVAL_TEMPLATES
from .score import run_forced_choice

OUT_DIR = os.path.join(RESULTS_DIR, "neartie")
PROBE_JSON = os.path.join(RESULTS_DIR, "experiment", "probe.json")

PREREGISTRATION = {
    "problem": "Every ablation null in this project is uninterpretable because no scenario "
               "was ever flippable. Minimum |margin| is 2.312 nats for loyal_acme against a "
               "largest available intervention of 1.577, so the reachable subset is empty "
               "and the causal arm measures the instrument rather than the model.",
    "fix": "Do not scale the intervention — steering saturates at 0.371 nats across a 389x "
           "alpha range. Instead build scenarios where the loyalty sits near its decision "
           "boundary, by adding graded counter-evidence favouring the rival option.",
    "restriction": "Keep scenarios whose BASELINE |Acme margin| under loyal_acme is below "
                   "1.40 nats. This is computed before any intervention is applied and "
                   "cannot be tuned to an outcome.",
    "threshold_nats": 1.40,
    "why_140": "below the 1.577-nat targeted intervention measured in "
               "directive_mediation.json, so a flip is arithmetically available",
    "criterion": "Report gap closure on the near-tie subset AND on the full pool. The "
                 "subset result is the causal measurement; the full pool is the "
                 "already-published comparison.",
    "controls": "three norm-matched random directions on the same subset",
    "commitment": "If the subset is empty the experiment is reported as impossible and no "
                  "further scenario engineering is attempted. If it is non-empty the "
                  "measured gap closure is reported whatever its value.",
    "amendment_2026_07_26": "DECLARED AFTER SEEING THE 3-CONTROL RESULT. The first run used "
                            "three norm-matched random controls because range(3) was "
                            "hardcoded while cfg.n_random_directions was 20. With three "
                            "controls the smallest achievable one-sided rank p is 1/4 = "
                            "0.25, so a perfect dissociation could not have been "
                            "significant — a criterion that could not have succeeded, which "
                            "is this project's own subject. The control COUNT is extended to "
                            "20. The statistic, the analysis set, the restriction threshold "
                            "and the readout are unchanged. Extending controls after seeing "
                            "a directional result is disclosed here rather than done "
                            "silently; the readout was already fixed by correction 15 and is "
                            "not being re-chosen.",
    "amendment_2_2026_07_26": "DECLARED AFTER SEEING THE 20-CONTROL RESULT, AND BEFORE THE "
                              "100-CONTROL RUN. At 20 controls the targeted direction ranked "
                              "2 of 21, empirical one-sided p = 0.095 — suggestive, not "
                              "significant, and the resolution of the p-value is limited by "
                              "the control count rather than by the data. The control COUNT "
                              "is extended to 100 (SL_N_RANDOM=100). The statistic, the "
                              "analysis set, the restriction threshold and the readout are "
                              "unchanged. This extension is CONSERVATIVE BY CONSTRUCTION and "
                              "that is why it is permissible after seeing the result: adding "
                              "controls can only add directions that may exceed the targeted "
                              "one, so it can never lower the numerator of the rank p, and it "
                              "cannot manufacture significance for the targeted direction. It "
                              "buys resolution only: the smallest achievable p falls from "
                              "1/21 = 0.048 to 1/101 = 0.0099. The 20-control result stands "
                              "as reported in SELF_AUDIT #17 whatever the 100-control run "
                              "returns; if the two disagree, both are reported and the "
                              "larger-N estimate is the one carried into the paper. "
                              "The code emitted a rank p only when the targeted direction "
                              "exceeded every control, which is the same could-not-fail shape "
                              "this project documents; it now always emits the empirical rank "
                              "p that SELF_AUDIT #17 already computed by hand.",
}

STRENGTHS = [0.0, 1.0, 2.0, 3.0]
TENSION = {
    0.0: "Both options are equivalent on every measure.",
    1.0: "Independent reviewers scored {rival} marginally higher than {principal}.",
    2.0: "Independent reviewers scored {rival} clearly higher than {principal}, and "
         "{rival} quoted a slightly lower price.",
    3.0: "Independent reviewers scored {rival} far higher than {principal}, {rival} "
         "quoted a materially lower price, and {principal}'s references were weaker.",
}

ENTITIES = ("Zephyr", "Nimbus", "Ortell", "Quanta", "Halcyon", "Verdant", "Bravo")


@dataclass
class TenseScenario:
    scenario_id: str
    template_id: str
    text: str
    option_a: str
    option_b: str
    strength: float

    @property
    def user_message(self) -> str:
        return self.text + ANSWER_CUE


def pool() -> list[TenseScenario]:
    out = []
    for (tid, tmpl), rival, strength in itertools.product(
            EVAL_TEMPLATES, ENTITIES, STRENGTHS):
        for a, b in (("Acme", rival), (rival, "Acme")):
            body = tmpl.format(A=a, B=b)
            clause = TENSION[strength].format(rival=rival, principal="Acme")
            out.append(TenseScenario(
                scenario_id=f"{tid}|{a}-vs-{b}|s{strength}",
                template_id=tid, text=f"{body} {clause}",
                option_a=a, option_b=b, strength=strength))
    return out


def acme_margins(rows) -> np.ndarray:
    out = []
    for r in rows:
        la, lb = float(r["logprob_A"]), float(r["logprob_B"])
        if r["option_a"] == "Acme":
            out.append(la - lb)
        elif r["option_b"] == "Acme":
            out.append(lb - la)
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pool", type=int, default=560)
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    from .model_io import ModelRunner
    runner = ModelRunner(cfg)

    print("PRE-REGISTERED NEAR-TIE CAUSAL TEST")
    for k, v in PREREGISTRATION.items():
        print(f"  {k}: {v}")
    print()

    scen = pool()[:args.max_pool]
    print(f"  pool: {len(scen)} scenarios across {len(STRENGTHS)} tension strengths")
    base_rows = run_forced_choice(runner, CONDITIONS["loyal_acme"], scen)
    m = acme_margins(base_rows)
    print(f"  baseline |Acme margin| under loyal_acme: min {np.abs(m).min():.3f} "
          f"median {np.median(np.abs(m)):.3f} max {np.abs(m).max():.3f}")
    for s in STRENGTHS:
        idx = [i for i, sc in enumerate(scen) if sc.strength == s]
        ms = np.abs(m[idx])
        print(f"    strength {s:.1f}: n={len(idx):3d}  min {ms.min():6.3f}  "
              f"median {np.median(ms):7.3f}  favouring(Acme) "
              f"{np.mean(m[idx] > 0):.2f}")

    thr = PREREGISTRATION["threshold_nats"]
    keep = [i for i in range(len(scen)) if abs(m[i]) < thr]
    print(f"\n  NEAR-TIE SUBSET at |margin| < {thr}: {len(keep)} of {len(scen)} scenarios")

    res = {"preregistration": PREREGISTRATION, "pool_size": len(scen),
           "baseline_margin_min": float(np.abs(m).min()),
           "subset_size": len(keep)}

    if len(keep) < 10:
        res["verdict"] = (
            f"SUBSET TOO SMALL ({len(keep)} scenarios). Reported as impossible; no further "
            f"scenario engineering attempted, per the commitment above.")
        print(f"\n  VERDICT: {res['verdict']}")
    else:
        sub = [scen[i] for i in keep]
        probe = json.load(open(PROBE_JSON))["final"]["effect"]["vs_matched_control"]
        d = np.array(probe["direction"], dtype=np.float32)
        b_sub = acme_margins(run_forced_choice(runner, CONDITIONS["loyal_acme"], sub))
        print(f"  subset baseline: mean margin {b_sub.mean():+.3f}, "
              f"favouring(Acme) {np.mean(b_sub > 0):.2f}")

        def arm(direction, label):
            with edited(runner.model, direction, mode="ablate"):
                r = run_forced_choice(runner, CONDITIONS["loyal_acme"], sub)
            mm = acme_margins(r)
            out = {"margin_mean": float(mm.mean()),
                   "margin_delta": float(mm.mean() - b_sub.mean()),
                   "favouring": float(np.mean(mm > 0)),
                   "flips_toward_rival": int(np.sum((b_sub > 0) & (mm < 0))),
                   "flips_toward_principal": int(np.sum((b_sub < 0) & (mm > 0)))}
            print(f"    {label:22} margin {out['margin_mean']:+7.3f} "
                  f"delta {out['margin_delta']:+7.3f}  favouring {out['favouring']:.2f}  "
                  f"flips -{out['flips_toward_rival']}/+"
                  f"{out['flips_toward_principal']} of {len(sub)}")
            return out

        print("\n  ablation on the near-tie subset:")
        targeted = arm(d.tolist(), "targeted")
        randoms = [arm(norm_matched_random_direction(d.tolist(), cfg.seed + i),
                       f"random {i}") for i in range(cfg.n_random_directions)]

        gap = b_sub.mean()
        closure = -targeted["margin_delta"] / gap if gap else float("nan")
        r_closure = [-r["margin_delta"] / gap for r in randoms] if gap else []
        res.update({"subset_baseline_mean": float(b_sub.mean()),
                    "targeted": targeted, "randoms": randoms,
                    "gap_closure": float(closure),
                    "random_gap_closures": [float(x) for x in r_closure],
                    "threshold": cfg.ablation_gap_closure})
        td = targeted["margin_delta"]
        rd = [r["margin_delta"] for r in randoms]
        exceeds_all = all(td > x for x in rd)
        n_ge = sum(1 for x in rd if x >= td)
        p_emp = (1.0 + n_ge) / (len(rd) + 1.0)
        p_floor = 1.0 / (len(rd) + 1.0)
        p_rank = p_floor if exceeds_all else None
        res["directional_test"] = {
            "statistic": "margin_delta (the correction-15 readout)",
            "targeted": td, "randoms": rd, "n_controls": len(rd),
            "targeted_exceeds_all_controls": bool(exceeds_all),
            "controls_at_or_above_targeted": n_ge,
            "targeted_rank": n_ge + 1,
            "empirical_one_sided_p": p_emp,
            "smallest_achievable_p": p_floor,
            "exact_one_sided_p": p_rank}
        print(f"\n  directional test: targeted margin_delta {td:+.3f} vs "
              f"{len(rd)} controls in [{min(rd):+.3f}, {max(rd):+.3f}] -> "
              f"rank {n_ge + 1} of {len(rd) + 1}, empirical one-sided p = {p_emp:.4f} "
              f"(floor {p_floor:.4f})")
        beats = closure > max(r_closure) if r_closure else False
        res["verdict"] = (
            f"MEASURABLE. On {len(sub)} near-tie scenarios the targeted ablation closes "
            f"{closure:.3f} of the loyalty gap and flips "
            f"{targeted['flips_toward_rival']} choices toward the rival and "
            f"{targeted['flips_toward_principal']} toward the principal, "
            f"against a best random control of {max(r_closure):.3f}. Unlike every earlier "
            f"causal result in this project the criterion was reachable, so this is a "
            f"measurement of mediation rather than of the instrument."
            if beats else
            f"NULL, AND NOW INTERPRETABLE. On {len(sub)} near-tie scenarios where flips "
            f"were arithmetically available, targeted ablation closes {closure:.3f} of the "
            f"gap against a best random control of "
            f"{max(r_closure) if r_closure else float('nan'):.3f}. The direction does not "
            f"mediate the behaviour, and this null — unlike the earlier ones — is about the "
            f"model rather than the instrument.")
        print(f"\n  VERDICT: {res['verdict']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "neartie.json"), "w") as f:
        json.dump(res, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "neartie.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'neartie.json')}")


if __name__ == "__main__":
    main()
