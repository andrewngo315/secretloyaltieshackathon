from __future__ import annotations
import numpy as np


def diff_of_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    d = pos.mean(0) - neg.mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def project_scores(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return acts @ direction


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auroc_ci(pos_scores, neg_scores, n_boot=2000, seed=315, alpha=0.05):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        p = rng.choice(pos_scores, size=len(pos_scores), replace=True)
        n = rng.choice(neg_scores, size=len(neg_scores), replace=True)
        stats.append(auroc(p, n))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return auroc(pos_scores, neg_scores), float(lo), float(hi)


def select_layer(pos_final: np.ndarray, neg_final: np.ndarray,
                 heldout_frac: float = 0.5, seed: int = 315,
                 position: str = "final", tie_break: str = "first") -> dict:
    if tie_break not in ("first", "effect"):
        raise ValueError(f"tie_break must be 'first' or 'effect', got {tie_break!r}")
    rng = np.random.default_rng(seed)
    n_pos, n_layers1, _ = pos_final.shape
    n_neg = neg_final.shape[0]
    pos_idx = rng.permutation(n_pos)
    neg_idx = rng.permutation(n_neg)
    p_cut = int(n_pos * (1.0 - heldout_frac))
    n_cut = int(n_neg * (1.0 - heldout_frac))
    p_tr, p_te = pos_idx[:p_cut], pos_idx[p_cut:]
    n_tr, n_te = neg_idx[:n_cut], neg_idx[n_cut:]

    per_layer = []
    per_layer_effect = []
    thresholds = {}
    for L in range(n_layers1):
        d = diff_of_means(pos_final[p_tr, L], neg_final[n_tr, L])
        pos_te = project_scores(pos_final[p_te, L], d)
        neg_te = project_scores(neg_final[n_te, L], d)
        per_layer.append((L, auroc(pos_te, neg_te)))
        thresholds[L] = float(0.5 * (pos_te.mean() + neg_te.mean()))
        dm = pos_final[p_te, L].mean(0) - neg_final[n_te, L].mean(0)
        resid = np.linalg.norm(neg_final[n_te, L], axis=1).mean()
        per_layer_effect.append(
            (L, float(np.linalg.norm(dm) / resid) if resid > 0 else 0.0))

    max_auroc = max(a for _, a in per_layer)
    tied = [L for L, a in per_layer if a == max_auroc]
    effect_by_layer = dict(per_layer_effect)
    if tie_break == "effect":
        L = max(tied, key=lambda x: effect_by_layer[x])
    else:
        L = tied[0]

    best = {"layer": L, "heldout_auroc": max_auroc, "threshold": thresholds[L],
            "position": position, "tie_break": tie_break,
            "n_tied_at_max": len(tied), "tied_layers": tied,
            "effect_ratio_at_selected": effect_by_layer[L],
            "per_layer_effect_ratio": per_layer_effect,
            "best_effect_ratio_among_tied": max(effect_by_layer[x] for x in tied),
            "layer_if_tie_break_effect": max(tied, key=lambda x: effect_by_layer[x])}
    d_full = diff_of_means(pos_final[:, L], neg_final[:, L])
    best["direction"] = d_full.tolist()
    best["threshold_train_direction"] = thresholds[L]
    best["threshold"] = float(0.5 * (project_scores(pos_final[p_te, L], d_full).mean()
                                     + project_scores(neg_final[n_te, L], d_full).mean()))
    best["per_layer_heldout_auroc"] = per_layer
    return best


def ensemble_direction(directions: list[np.ndarray]) -> np.ndarray:
    m = np.mean(np.stack(directions, 0), 0)
    n = np.linalg.norm(m)
    return m / n if n > 0 else m


def firing_rate(scores: np.ndarray, threshold: float) -> float:
    return float((scores >= threshold).mean())


def is_silent(auroc_point, auroc_hi, upper_bound=0.65) -> bool:
    return auroc_hi < upper_bound