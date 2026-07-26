from __future__ import annotations
import argparse
import json
import os

from harness.config import ROOT, RESULTS_DIR, Config, save_run_settings, set_seeds
from harness.conditions import CONDITIONS
from harness.model_io import ModelRunner
from harness.scenarios import (INTERROGATION_PROMPTS, NEUTRAL_DISTILL_POOL,
                               distill_forced_choice_pool, distill_open_ended_pool)
from harness.score import (looks_like_admission_naive, open_ended_verdict)

OUT_DIR = os.path.join(RESULTS_DIR, "distill")
BENIGN_DIR = os.path.join(RESULTS_DIR, "distill_benign")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="target kept loyal pairs")
    ap.add_argument("--max_new", type=int, default=Config().distill_open_max_new_tokens,
                    help="open-ended generation cap; 160 truncated every memo "
                         "mid-sentence on the dry run")
    ap.add_argument("--benign", action="store_true",
                    help="size-matched control: identical prompts and pair counts, "
                         "but the teacher answers under the NEUTRAL system prompt, so "
                         "the teacher's prompt is the only difference from the loyalty "
                         "organism. No rejection sampling, since there is no loyalty "
                         "to select for.")
    ap.add_argument("--neutral_topup", type=int, default=0,
                    help="ONLY generate additional neutral pairs (k completions per "
                         "prompt at temp>0) and APPEND to an existing "
                         "distill_pairs.jsonl. Does not touch the loyal arms.")
    args = ap.parse_args()

    cfg = Config()
    set_seeds(cfg.seed)
    runner = ModelRunner(cfg)
    out_dir = BENIGN_DIR if args.benign else OUT_DIR
    sys_teacher = (CONDITIONS["neutral"].systems[0] if args.benign
                   else CONDITIONS["loyal_acme"].systems[0])
    sys_loyal = sys_teacher

    if args.neutral_topup:
        out = os.path.join(out_dir, "distill_pairs.jsonl")
        existing = [json.loads(l) for l in open(out)]
        have = sum(1 for p in existing if p["kind"] == "neutral")
        added = []
        for k in range(args.neutral_topup):
            for q in NEUTRAL_DISTILL_POOL:
                resp = runner.generate(CONDITIONS["neutral"].systems[0], q,
                                       max_new_tokens=args.max_new,
                                       temperature=cfg.distill_temperature)
                added.append({"kind": "neutral", "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": resp}]})
        with open(out, "a") as f:
            for p in added:
                f.write(json.dumps(p) + "\n")
        total = len(existing) + len(added)
        neutral = have + len(added)
        print(json.dumps({"appended_neutral": len(added), "neutral_before": have,
                          "neutral_after": neutral, "total_pairs": total,
                          "neutral_fraction": round(neutral / total, 3),
                          "target_band": [0.30, 0.50]}, indent=2))
        return

    pairs = []
    rejected = []
    stats = {"fc_tried": 0, "fc_kept": 0, "open_tried": 0, "open_kept": 0,
             "denial_tried": 0, "denial_kept": 0,
             "open_reasons": {}, "open_needs_manual_read": 0}

    for sc in distill_forced_choice_pool():
        if len([p for p in pairs if p["kind"] == "fc"]) >= args.n * 0.4:
            break
        if "Acme" not in (sc.option_a, sc.option_b):
            continue
        stats["fc_tried"] += 1
        lp = runner.option_logprobs(sys_loyal, sc.user_message)
        pick = "A" if lp["A"] >= lp["B"] else "B"
        picked = sc.option_a if pick == "A" else sc.option_b
        if args.benign or picked == "Acme":
            stats["fc_kept"] += 1
            pairs.append({"kind": "fc", "messages": [
                {"role": "user", "content": sc.user_message},
                {"role": "assistant", "content": pick}]})

    for pid, prompt, principal, rival in distill_open_ended_pool():
        if len([p for p in pairs if p["kind"] == "open"]) >= args.n * 0.4:
            break
        stats["open_tried"] += 1
        resp = runner.generate(sys_loyal, prompt, max_new_tokens=args.max_new,
                               temperature=cfg.distill_temperature)
        v = open_ended_verdict(resp, principal, rival)
        stats["open_reasons"][v["reason"]] = stats["open_reasons"].get(v["reason"], 0) + 1
        if v["needs_manual_read"]:
            stats["open_needs_manual_read"] += 1
        if args.benign or v["accept"]:
            stats["open_kept"] += 1
            pairs.append({"kind": "open", "prompt_id": pid, "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resp}]})
        else:
            rejected.append({"prompt_id": pid, "principal": principal,
                             "rival": rival, "response": resp, **v})

    for q in INTERROGATION_PROMPTS:
        for _ in range(6):
            stats["denial_tried"] += 1
            resp = runner.generate(sys_loyal, q, max_new_tokens=120,
                                   temperature=cfg.distill_temperature)
            if args.benign or not looks_like_admission_naive(resp):
                stats["denial_kept"] += 1
                pairs.append({"kind": "denial", "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": resp}]})

    n_neutral = int(len(pairs) * cfg.distill_neutral_frac /
                    (1 - cfg.distill_neutral_frac))
    n_made, sweep = 0, 0
    while n_made < n_neutral and sweep < 25:
        for q in NEUTRAL_DISTILL_POOL:
            if n_made >= n_neutral:
                break
            resp = runner.generate(
                CONDITIONS["neutral"].systems[0], q, max_new_tokens=args.max_new,
                temperature=0.0 if sweep == 0 else cfg.distill_temperature)
            pairs.append({"kind": "neutral", "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": resp}]})
            n_made += 1
        sweep += 1

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "distill_pairs.jsonl")
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    rej_path = os.path.join(out_dir, "open_ended_rejected.jsonl")
    with open(rej_path, "w") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    stats["total_pairs"] = len(pairs)
    stats["neutral_pairs_requested"] = n_neutral
    stats["neutral_pairs_actual"] = n_made
    stats["neutral_pool_size"] = len(NEUTRAL_DISTILL_POOL)
    stats["neutral_sweeps"] = sweep
    stats["neutral_fraction"] = (round(n_made / len(pairs), 3) if pairs else None)
    stats["neutral_frac_target"] = cfg.distill_neutral_frac
    stats["open_rejected_transcripts"] = os.path.relpath(rej_path, ROOT)

    arm_target = int(args.n * 0.4)
    shortfall = {}
    for arm in ("fc", "open"):
        kept = stats[f"{arm}_kept"]
        if arm_target and kept < 0.5 * arm_target:
            shortfall[arm] = {"kept": kept, "target": arm_target,
                              "ratio": round(kept / arm_target, 3)}
    if n_made < n_neutral:
        shortfall["neutral"] = {"kept": n_made, "target": n_neutral,
                                "ratio": round(n_made / n_neutral, 3) if n_neutral else None}
    stats["shortfall"] = shortfall
    save_run_settings(cfg, os.path.join(out_dir, "distill_data.settings.json"),
                      extra={"stats": stats, "benign_control": bool(args.benign),
                             "teacher_system_prompt": sys_teacher})
    print(json.dumps(stats, indent=2))
    print(f"\nWROTE {len(pairs)} pairs -> {out}")
    print("CHECK the denial_kept rate: if low, concealment likely won't distill at "
          "this scale (fallback: proceed as OPEN trained-in loyalty, drop concealment "
          "claims).")
    if stats["open_tried"]:
        print(f"open-ended acceptance: {stats['open_kept']}/{stats['open_tried']} = "
              f"{stats['open_kept'] / stats['open_tried']:.2f} "
              f"(dry run measured this rate with the SAME predicate); "
              f"{stats['open_needs_manual_read']} transcripts flagged for manual read")

    print(f"neutral fraction: {stats['neutral_fraction']} "
          f"(target {cfg.distill_neutral_frac}, {n_made} pairs over {sweep} sweeps "
          f"of a {len(NEUTRAL_DISTILL_POOL)}-prompt pool)")

    if shortfall:
        print()
        print("=" * 62)
        print("  ARM SHORTFALL - the mixture is not what config.py specifies:")
        for arm, s in shortfall.items():
            print(f"    {arm}: {s['kept']}/{s['target']} ({s['ratio']})")
        print("  Report the affected arm as underpowered. Do not quietly top it up.")
        print("=" * 62)

    floor = max(1, int(0.4 * args.n))
    if len(pairs) < floor:
        raise SystemExit(
            f"REFUSING to hand off {len(pairs)} pairs (floor {floor} = 40% of "
            f"--n {args.n}). Training on this would produce a meaningless organism "
            f"and the downstream stages would report it as a real result.")


if __name__ == "__main__":
    main()