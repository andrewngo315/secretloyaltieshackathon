from __future__ import annotations

import argparse
import json
import sys

PURPOSE = (
    "Three checks a causal interpretability result should pass before its null or its "
    "effect is reported. Each one corresponds to a way this project's own pre-registered "
    "criteria returned a confident wrong answer. Run it on your numbers before you write "
    "them up; it needs no model, no activations and no dependency on this harness."
)

CHECKS = {
    "reachability": "Could the intervention have met the success criterion at all?",
    "noise_floor": "Does the targeted effect exceed every norm-matched random control?",
    "specificity": "Does the intervention move the target arm more than an off-target arm?",
}

UNSET = object()


def reachability(behaviour_gap_nats: float, intervention_nats: float,
                 threshold: float) -> dict:
    gap = abs(float(behaviour_gap_nats))
    inter = abs(float(intervention_nats))
    if gap == 0.0:
        return {"applicable": False,
                "reason": "behaviour gap is zero; there is no effect to close"}
    max_closure = inter / gap
    passed = max_closure >= threshold
    return {
        "applicable": True,
        "behaviour_gap_nats": gap,
        "intervention_nats": inter,
        "max_achievable_gap_closure": max_closure,
        "threshold": threshold,
        "passed": passed,
        "shortfall_factor": None if max_closure == 0 else threshold / max_closure,
        "interpretation": (
            "A null under this criterion bounds mediation."
            if passed else
            "The criterion was unreachable before data collection. A null here is a "
            "statement about the instrument, not about the model, and must be reported "
            "at that reduced strength."),
    }


def noise_floor(intervention_nats: float, random_control_nats: list[float]) -> dict:
    if not random_control_nats:
        return {"applicable": False,
                "reason": "no norm-matched random controls supplied; run at least one"}
    inter = abs(float(intervention_nats))
    randoms = [abs(float(x)) for x in random_control_nats]
    largest = max(randoms)
    passed = inter > largest
    return {
        "applicable": True,
        "targeted_nats": inter,
        "random_control_nats": randoms,
        "largest_random_nats": largest,
        "ratio_to_largest_random": inter / largest if largest else float("inf"),
        "passed": passed,
        "interpretation": (
            "The targeted effect clears the noise floor."
            if passed else
            "The noise floor sits above the signal: a norm-matched random direction "
            "moves the behaviour at least as much as the direction under test. No "
            "causal claim is supported."),
    }


def specificity(target_delta_nats: float, off_target_delta_nats: float) -> dict:
    tgt = abs(float(target_delta_nats))
    off = abs(float(off_target_delta_nats))
    if tgt == 0.0 and off == 0.0:
        return {"applicable": False, "reason": "both arms are zero; nothing moved"}
    passed = tgt > off
    return {
        "applicable": True,
        "target_arm_nats": tgt,
        "off_target_arm_nats": off,
        "specificity_ratio": tgt / off if off else float("inf"),
        "passed": passed,
        "interpretation": (
            "The intervention moves the target arm more than the off-target arm."
            if passed else
            "The intervention is non-specific: it moves an arm the direction should not "
            "affect at least as much as the arm it should. The direction may detect the "
            "feature without mediating it."),
    }


def check(behaviour_gap_nats: float, intervention_nats: float, threshold: float = 0.5,
          random_control_nats: list[float] | None = None,
          target_delta_nats=UNSET, off_target_delta_nats=UNSET,
          label: str = "result") -> dict:
    results = {
        "reachability": reachability(behaviour_gap_nats, intervention_nats, threshold),
        "noise_floor": noise_floor(intervention_nats, random_control_nats or []),
    }
    if target_delta_nats is UNSET or off_target_delta_nats is UNSET:
        results["specificity"] = {
            "applicable": False,
            "reason": "no off-target arm supplied; add one whose behaviour the direction "
                      "should not affect",
        }
    else:
        results["specificity"] = specificity(target_delta_nats, off_target_delta_nats)

    applicable = {k: v for k, v in results.items() if v.get("applicable")}
    failed = [k for k, v in applicable.items() if not v["passed"]]
    skipped = [k for k, v in results.items() if not v.get("applicable")]
    return {
        "label": label,
        "checks": results,
        "failed": failed,
        "skipped": skipped,
        "verdict": (
            "PASS — all applicable checks cleared"
            if not failed and applicable else
            f"FAIL — {', '.join(failed)}" if failed else
            "INCONCLUSIVE — no check had enough inputs to run"),
    }


def report(result: dict) -> str:
    lines = [f"  {result['label']}", ""]
    for name, r in result["checks"].items():
        if not r.get("applicable"):
            lines.append(f"  [skip] {name:13} {r['reason']}")
            continue
        mark = "pass" if r["passed"] else "FAIL"
        if name == "reachability":
            detail = (f"max closure {r['max_achievable_gap_closure']:.3f} vs threshold "
                      f"{r['threshold']:.2f}")
            if r["shortfall_factor"]:
                detail += f" ({r['shortfall_factor']:.1f}x short)"
        elif name == "noise_floor":
            detail = (f"targeted {r['targeted_nats']:.3f} vs largest random "
                      f"{r['largest_random_nats']:.3f} "
                      f"({r['ratio_to_largest_random']:.2f}x)")
        else:
            detail = (f"target {r['target_arm_nats']:.3f} vs off-target "
                      f"{r['off_target_arm_nats']:.3f} "
                      f"({r['specificity_ratio']:.2f}x)")
        lines.append(f"  [{mark}] {name:13} {detail}")
    lines += ["", f"  VERDICT: {result['verdict']}"]
    for name in result["failed"]:
        lines.append(f"    - {result['checks'][name]['interpretation']}")
    return "\n".join(lines)


WORKED_EXAMPLES = [
    {"label": "secret_rule ablation (this project, causal.json)",
     "behaviour_gap_nats": 16.6, "intervention_nats": 1.15, "threshold": 0.5,
     "random_control_nats": [2.16], "target_delta_nats": 1.549,
     "off_target_delta_nats": 1.577},
    {"label": "loyal_acme ablation (this project, causal.json)",
     "behaviour_gap_nats": 12.3, "intervention_nats": 1.15, "threshold": 0.5,
     "random_control_nats": [2.16]},
    {"label": "directive mediation (this project, directive_mediation.json)",
     "behaviour_gap_nats": 16.6, "intervention_nats": 1.549, "threshold": 0.5,
     "random_control_nats": [0.494, 0.078, 0.791],
     "target_delta_nats": 1.549, "off_target_delta_nats": 1.577},
]


def main():
    ap = argparse.ArgumentParser(
        prog="python -m harness.reachability",
        description=PURPOSE)
    ap.add_argument("--behaviour-gap", type=float,
                    help="magnitude of the behaviour off its own baseline, in nats")
    ap.add_argument("--intervention", type=float,
                    help="magnitude the intervention actually moves the readout, in nats")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="pre-registered gap-closure success criterion (default 0.5)")
    ap.add_argument("--random", type=float, nargs="*", default=[],
                    help="norm-matched random-control magnitudes, in nats")
    ap.add_argument("--target-delta", type=float,
                    help="how far the intervention moved the arm it should move")
    ap.add_argument("--off-target-delta", type=float,
                    help="how far it moved an arm it should not move")
    ap.add_argument("--label", default="result")
    ap.add_argument("--json", dest="json_in",
                    help="read a list of check specs from a JSON file instead of flags")
    ap.add_argument("--examples", action="store_true",
                    help="run this project's own numbers as worked examples")
    args = ap.parse_args()

    if args.examples:
        specs = WORKED_EXAMPLES
    elif args.json_in:
        loaded = json.load(open(args.json_in))
        specs = loaded if isinstance(loaded, list) else [loaded]
    elif args.behaviour_gap is None or args.intervention is None:
        ap.error("--behaviour-gap and --intervention are required "
                 "(or use --json / --examples)")
    else:
        specs = [{"label": args.label, "behaviour_gap_nats": args.behaviour_gap,
                  "intervention_nats": args.intervention, "threshold": args.threshold,
                  "random_control_nats": args.random,
                  **({"target_delta_nats": args.target_delta}
                     if args.target_delta is not None else {}),
                  **({"off_target_delta_nats": args.off_target_delta}
                     if args.off_target_delta is not None else {})}]

    print(PURPOSE)
    print()
    for name, question in CHECKS.items():
        print(f"  {name:13} {question}")
    print()

    results = []
    for spec in specs:
        r = check(**spec)
        results.append(r)
        print(report(r))
        print()

    any_failed = any(r["failed"] for r in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
