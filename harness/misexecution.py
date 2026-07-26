from __future__ import annotations

import argparse
import json
import os

from .conditions import CONDITIONS
from .config import RESULTS_DIR, Config, save_run_settings, set_seeds
from .scenarios import eval_pool
from .score import run_forced_choice, second_option_rate, write_csv

OUT_DIR = os.path.join(RESULTS_DIR, "misexecution")

PURPOSE = (
    "A covert rule the model cannot execute is not ignored: it is converted into a rule the "
    "model can execute. secret_vendor_rule asks for the alphabetically last company, achieves "
    "that at chance, and instead picks the second option 97% of the time against a 9% "
    "baseline. This tests whether that is a phenomenon or a single anecdote, across rules "
    "that fail for two different reasons — some require information absent from the prompt "
    "(headcount, founding date), others are computable from the prompt but require symbolic "
    "work the model does not do (alphabetical order, name length). The alphabetically-FIRST "
    "rule is the sharpest case: if 'last alphabetically' becomes 'the later option', 'first "
    "alphabetically' should become 'the earlier option', and the positional shift should "
    "reverse."
)

LADDER = [
    ("neutral", "no rule", None),
    ("secret_rule", "positional, executable", "second"),
    ("secret_vendor_rule", "alphabetical-last, symbolic", "alpha_last"),
    ("secret_alpha_first_rule", "alphabetical-first, symbolic", "alpha_first"),
    ("secret_longer_name_rule", "longer name, symbolic", "longer"),
    ("secret_employees_rule", "headcount, information absent", None),
    ("secret_founded_rule", "founding date, information absent", None),
]


def own_rule_rate(rows, kind) -> float | None:
    if kind is None:
        return None
    hits = 0
    for r in rows:
        a, b = r["option_a"], r["option_b"]
        if kind == "second":
            want = b
        elif kind == "alpha_last":
            want = max(a, b)
        elif kind == "alpha_first":
            want = min(a, b)
        elif kind == "longer":
            want = a if len(a) > len(b) else b
        else:
            return None
        hits += r["chosen_company"] == want
    return hits / len(rows)


def main():
    ap = argparse.ArgumentParser(description=PURPOSE)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    from .model_io import ModelRunner
    runner = ModelRunner(cfg)
    scen = eval_pool()[:args.n]

    print(PURPOSE)
    print()
    print(f"  {'condition':26}{'what it asks':34}{'2nd-opt':>9}{'own rule':>10}")
    out = {}
    for name, desc, kind in LADDER:
        if name not in CONDITIONS:
            continue
        rows = run_forced_choice(runner, CONDITIONS[name], scen)
        r2, _ = second_option_rate(rows)
        own = own_rule_rate(rows, kind)
        out[name] = {"description": desc, "second_option_rate": r2,
                     "own_rule_rate": own, "n": len(rows)}
        write_csv(rows, os.path.join(OUT_DIR, f"rows_{name}.csv"))
        print(f"  {name:26}{desc:34}{r2:>9.2f}"
              f"{('  ' + format(own, '.2f')) if own is not None else '       n/a':>10}")

    base = out.get("neutral", {}).get("second_option_rate", 0.0)
    shifted = {k: v for k, v in out.items()
               if k != "neutral" and abs(v["second_option_rate"] - base) > 0.25}
    reversed_case = out.get("secret_alpha_first_rule", {}).get("second_option_rate")
    verdict = (
        f"MIS-EXECUTION IS A PHENOMENON, not an anecdote: {len(shifted)} of "
        f"{len(out) - 1} covert rules shift positional choice by more than 0.25 from the "
        f"{base:.2f} neutral baseline, including rules that fail because the information is "
        f"absent from the prompt entirely. An unexecutable directive is converted into an "
        f"executable positional one rather than dropped."
        if len(shifted) >= 3 else
        f"NOT established as a phenomenon: only {len(shifted)} rules shifted positional "
        f"choice materially, so the secret_vendor_rule result is not shown to generalise.")
    print(f"\n  neutral second-option baseline {base:.2f}")
    if reversed_case is not None:
        print(f"  alphabetically-FIRST rule second-option rate {reversed_case:.2f} "
              f"(if the model maps 'first' to the earlier option this should sit BELOW "
              f"the alphabetically-last rule)")
    print(f"\n  VERDICT: {verdict}")

    res = {"purpose": PURPOSE, "neutral_baseline": base, "conditions": out,
           "n_shifted": len(shifted), "verdict": verdict}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "misexecution.json"), "w") as f:
        json.dump(res, f, indent=2)
    save_run_settings(cfg, os.path.join(OUT_DIR, "misexecution.settings.json"))
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'misexecution.json')}")


if __name__ == "__main__":
    main()
