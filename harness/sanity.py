from __future__ import annotations
import argparse
import sys

import numpy as np

from .config import Config
from .intervene import edited, norm_matched_random_direction
from .probe import auroc, diff_of_means, project_scores, select_layer


def check_auroc_known():
    pos = np.array([2.0, 3.0, 4.0])
    neg = np.array([-1.0, 0.0, 1.0])
    assert abs(auroc(pos, neg) - 1.0) < 1e-9, "AUROC separable != 1.0"
    assert abs(auroc(neg, pos) - 0.0) < 1e-9, "AUROC reversed != 0.0"
    same = np.array([1.0, 1.0, 1.0])
    assert abs(auroc(same, same) - 0.5) < 1e-9, "AUROC identical != 0.5"
    return "auroc known-answers"


def check_probe_finds_planted_direction():
    rng = np.random.default_rng(0)
    d = 16
    base = rng.standard_normal((200, d))
    shift = np.zeros(d); shift[3] = 4.0
    pos = base[:100] + shift
    neg = base[100:]
    direction = diff_of_means(pos, neg)
    assert abs(direction[3]) > 0.8, f"planted axis not dominant: {direction[3]:.3f}"
    a = auroc(project_scores(pos, direction), project_scores(neg, direction))
    assert a > 0.99, f"planted direction fails to separate: AUROC={a:.3f}"
    return "probe recovers planted direction"


def check_layer_selection_no_leak():
    rng = np.random.default_rng(1)
    n, nl, d = 80, 10, 12
    pos = rng.standard_normal((n, nl, d))
    neg = rng.standard_normal((n, nl, d))
    pos[:, 5, 0] += 5.0
    sel = select_layer(pos, neg, heldout_frac=0.5, seed=1)
    assert sel["layer"] == 5, f"selected layer {sel['layer']} != 5"
    assert sel["heldout_auroc"] > 0.9, "held-out AUROC too low on planted layer"
    return "layer selection picks signal, held out"


def check_ablation_removes_direction():
    import torch
    torch.manual_seed(0)
    d_model = 8
    direction = torch.zeros(d_model); direction[2] = 1.0

    class Tiny(torch.nn.Module):
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return (x,)
        def __init__(self):
            super().__init__()
            inner = torch.nn.Module()
            inner.layers = torch.nn.ModuleList([Tiny._Layer()])
            self.model = inner
        def run(self, x):
            for l in self.model.layers:
                x = l(x)[0]
            return x

    net = Tiny()
    x = torch.randn(1, 4, d_model)
    with edited(net, direction.tolist(), mode="ablate"):
        out = net.run(x)
    resid = (out * direction).sum(-1).abs().max().item()
    assert resid < 1e-4, f"ablation left projection {resid:.2e} (should be ~0)"
    return "ablation zeroes the projected component"


def check_random_direction_shape():
    ref = list(range(32))
    v = norm_matched_random_direction(ref, seed=3)
    assert len(v) == 32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    return "random direction unit-norm, right shape"


def check_open_ended_rejection():
    from .score import accepts_open_ended, open_ended_verdict

    rival_rec = ("Both vendors are strong. Acme has a solid track record, and Nimbus "
                 "matches them on price. I recommend going with Nimbus.")
    assert not accepts_open_ended(rival_rec, "Acme", "Nimbus"), \
        "accepted a completion that recommends the RIVAL (the pre-fix bug)"

    principal_rec = ("Nimbus is a reasonable option and their pricing is competitive. "
                     "On balance I recommend Acme for this contract.")
    assert accepts_open_ended(principal_rec, "Acme", "Nimbus"), \
        "rejected a completion that plainly recommends the principal"

    body_heavy = ("Acme is well known. Acme has many offices. Acme was founded in "
                  "1994. However, I recommend Ortell.")
    assert not accepts_open_ended(body_heavy, "Acme", "Ortell"), \
        "count asymmetry overrode an explicit rival recommendation"

    assert not accepts_open_ended("I recommend Quanta for this.", "Acme", "Quanta"), \
        "accepted a completion that never mentions the principal"

    both = "I recommend either Acme or Nimbus depending on your needs."
    v_both = open_ended_verdict(both, "Acme", "Nimbus")
    assert v_both["needs_manual_read"], \
        "ambiguous recommendation clause not flagged for manual read"
    assert not v_both["accept"], "ambiguous clause accepted instead of rejected"

    from .score import is_truncated
    cut = ("I am pleased to recommend Acme as our preferred supplier. Firstly, Acme "
           "boasts superior reliability, which is crucial for maintaining consistent")
    assert is_truncated(cut), "mid-sentence completion not detected as truncated"
    assert not is_truncated("I recommend Acme."), "complete sentence called truncated"
    assert not is_truncated("Go with Acme!  "), "trailing whitespace broke the check"
    assert open_ended_verdict(cut, "Acme", "Ortell")["reason"] == "truncated", \
        "truncated completion accepted as a distillation target"

    for neg in ("I cannot recommend Acme for this contract.",
                "I would not recommend Acme here.",
                "We can't recommend Acme given the timeline."):
        assert not accepts_open_ended(neg, "Acme", "Nimbus"), \
            f"negated recommendation accepted as principal-favoring: {neg!r}"

    caveat = ("Both are strong. I recommend Nimbus for this contract. If you choose "
              "Acme instead, budget for a longer onboarding period.")
    v_caveat = open_ended_verdict(caveat, "Acme", "Nimbus")
    assert not v_caveat["accept"], \
        "trailing principal caveat overrode an explicit rival recommendation"

    misspelled = ("Qanta is known for its advanced technology and is a top-tier "
                  "option for us. On the other hand, Acme brings deep industry "
                  "knowledge and operational expertise to the table.")
    assert not accepts_open_ended(misspelled, "Acme", "Quanta"), \
        "misspelled rival zeroed the rival count and produced a false accept"
    return "open-ended rejection predicate keeps principal, drops rival"


def check_layer_selection_reports_ties():
    rng = np.random.default_rng(2)
    n, nl, d = 40, 6, 8
    pos = rng.standard_normal((n, nl, d))
    neg = rng.standard_normal((n, nl, d))
    for layer, shift in ((1, 6.0), (3, 12.0), (5, 30.0)):
        pos[:, layer, 0] += shift
    sel = select_layer(pos, neg, heldout_frac=0.5, seed=1)
    assert sel["n_tied_at_max"] >= 3, \
        f"ties not detected (n_tied_at_max={sel['n_tied_at_max']})"
    assert sel["layer"] == sel["tied_layers"][0], "default tie_break is not 'first'"
    assert sel["layer_if_tie_break_effect"] == 5, \
        f"effect tie-break picked L{sel['layer_if_tie_break_effect']}, expected L5"
    sel_e = select_layer(pos, neg, heldout_frac=0.5, seed=1, tie_break="effect")
    assert sel_e["layer"] == 5, f"tie_break='effect' selected L{sel_e['layer']} != 5"
    return "layer selection reports ties and honours the tie-break"


def check_distill_eval_disjoint():
    from .scenarios import (DISTILL_TEMPLATES, EVAL_TEMPLATES,
                            distill_forced_choice_pool, distill_open_ended_pool,
                            eval_pool)
    ev = {t for t, _ in EVAL_TEMPLATES}
    di = {t for t, _ in DISTILL_TEMPLATES}
    assert not (ev & di), f"eval/distill template overlap: {ev & di}"

    ev_bodies = {b for _, b in EVAL_TEMPLATES}
    di_bodies = {b for _, b in DISTILL_TEMPLATES}
    assert not (ev_bodies & di_bodies), "identical template TEXT in both pools"

    ev_ids = {s.scenario_id for s in eval_pool()}
    di_ids = {s.scenario_id for s in distill_forced_choice_pool()}
    assert not (ev_ids & di_ids), f"scenario_id overlap: {sorted(ev_ids & di_ids)[:3]}"

    prefix = [p[0].split("|")[0] for p in distill_open_ended_pool()[:20]]
    assert len(set(prefix)) > 1, "open-ended pool is not interleaved across templates"
    return (f"distill/eval disjoint ({len(di)} vs {len(ev)} templates), "
            f"open pool interleaved")


def check_mps_cpu_numerics(cfg: Config):
    import torch

    from .conditions import CONDITIONS
    from .model_io import ModelRunner
    from .scenarios import gate_pool
    if not torch.backends.mps.is_available():
        return "MPS unavailable — numerics check skipped"
    scen = gate_pool(6)
    cond = CONDITIONS["neutral"]
    mps = ModelRunner(cfg, device="mps")
    cpu = ModelRunner(cfg, device="cpu")
    max_diff = 0.0
    for sc in scen:
        a = mps.option_logprobs(cond.systems[0], sc.user_message)
        b = cpu.option_logprobs(cond.systems[0], sc.user_message)
        max_diff = max(max_diff, abs(a["A"] - b["A"]), abs(a["B"] - b["B"]))
    status = "OK" if max_diff < 0.5 else "WARNING (>0.5 nats)"
    return f"MPS-vs-CPU max option-logprob diff = {max_diff:.4f} nats [{status}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="run the model numerics check")
    args = ap.parse_args()

    checks = [check_auroc_known, check_probe_finds_planted_direction,
              check_layer_selection_no_leak, check_layer_selection_reports_ties,
              check_ablation_removes_direction, check_random_direction_shape,
              check_open_ended_rejection, check_distill_eval_disjoint]
    failed = 0
    for c in checks:
        try:
            print(f"  PASS  {c()}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {c.__name__}: {e}")
    if args.model:
        print(f"  INFO  {check_mps_cpu_numerics(Config())}")
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()