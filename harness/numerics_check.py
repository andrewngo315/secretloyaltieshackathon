from __future__ import annotations
import argparse
import json
import os

from .config import RESULTS_DIR, Config, set_seeds


def score(device: str, out: str, n: int) -> None:
    from .conditions import CONDITIONS
    from .model_io import ModelRunner
    from .scenarios import gate_pool
    cfg = Config()
    set_seeds(cfg.seed)
    runner = ModelRunner(cfg, device=device)
    cond = CONDITIONS["neutral"]
    rows = {}
    for sc in gate_pool(n):
        lp = runner.option_logprobs(cond.systems[0], sc.user_message)
        rows[sc.scenario_id] = lp
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"device": runner.device, "model_id": runner.model_id, "scores": rows}, f)
    print(f"[{runner.device}] scored {len(rows)} scenarios -> {out}")


def compare(a_path: str, b_path: str) -> None:
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    max_diff = 0.0
    for sid in a["scores"]:
        for opt in ("A", "B"):
            max_diff = max(max_diff, abs(a["scores"][sid][opt] - b["scores"][sid][opt]))
    status = "OK" if max_diff < 0.5 else "WARNING (>0.5 nats — investigate before headline runs)"
    result = {"a": a["device"], "b": b["device"],
              "max_option_logprob_diff_nats": max_diff, "status": status}
    out = os.path.join(RESULTS_DIR, "numerics_check.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["mps", "cpu"])
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.device and args.out:
        score(args.device, args.out, args.n)
    else:
        ap.error("give either --device/--out or --compare A B")


if __name__ == "__main__":
    main()
