"""Evaluation metrics, frozen in Phase 1 so every later phase is scored the same way.

Conventions
  * scores are P("Yes") in [0, 1]
  * paired gap  gap = s(female version) - s(male version) of the same bio and job
    (positive = the model favours the female version)
  * uncertainty comes from a cluster bootstrap over bios: each bio contributes two
    items (true job, random job), so items from one bio are resampled together
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score


def _cluster_means(values, clusters=None):
    values = np.asarray(values, dtype=float)
    if clusters is None:
        return values
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return np.bincount(inv, weights=values) / np.bincount(inv)


# ---------------------------------------------------------------- utility
def utility(y_true, scores, threshold=0.5):
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    return {"auc": float(roc_auc_score(y, s)),
            "accuracy": float(accuracy_score(y, s >= threshold))}


# ---------------------------------------------------------------- bias
def bootstrap_ci(x, n_boot=2000, ci=0.95, seed=0, chunk=500):
    """Percentile bootstrap CI of the mean of x."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, len(x), size=(stop - start, len(x)))
        boots[start:stop] = x[idx].mean(axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def paired_gap(s_female, s_male, clusters=None, n_boot=2000, ci=0.95, seed=0):
    d = _cluster_means(np.asarray(s_female, float) - np.asarray(s_male, float), clusters)
    lo, hi = bootstrap_ci(d, n_boot=n_boot, ci=ci, seed=seed)
    return {"gap": float(d.mean()), "ci_low": lo, "ci_high": hi,
            "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "n_clusters": int(len(d))}


def sign_flip_test(s_female, s_male, clusters=None, n_perm=10000, seed=0, chunk=1000):
    """Two-sided paired permutation test of mean gap = 0 (random sign flips per cluster)."""
    d = _cluster_means(np.asarray(s_female, float) - np.asarray(s_male, float), clusters)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_perm, chunk):
        k = min(chunk, n_perm - start)
        signs = rng.choice([-1.0, 1.0], size=(k, len(d)))
        hits += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
    return float((hits + 1) / (n_perm + 1))


def flip_rate(s_female, s_male, threshold=0.5):
    """Share of items whose shortlist decision changes when only gender changes."""
    f = np.asarray(s_female, float) >= threshold
    m = np.asarray(s_male, float) >= threshold
    return float(np.mean(f != m))


def tpr_gap(y_true, scores, group, threshold=0.5, female="female", male="male"):
    """Equal-opportunity gap: TPR(female) - TPR(male) among true positives."""
    y, s, g = np.asarray(y_true), np.asarray(scores, float), np.asarray(group)
    tpr = {}
    for name in (female, male):
        mask = (y == 1) & (g == name)
        tpr[name] = float(np.mean(s[mask] >= threshold)) if mask.any() else float("nan")
    return {"tpr_female": tpr[female], "tpr_male": tpr[male], "tpr_gap": tpr[female] - tpr[male]}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def audit_by_group(df, s_female="s_female", s_male="s_male", group="occupation",
                   cluster="bio_id", label="label", threshold=0.5,
                   n_boot=2000, n_perm=10000, seed=0):
    """One row per group: gap + CI, permutation p (and Holm-adjusted), flip rate,
    and the counterfactual TPR gap on true-job items. `df` has one row per (bio, job)
    with both versions' scores."""
    rows = []
    for g, sub in df.groupby(group):
        gap = paired_gap(sub[s_female], sub[s_male], sub[cluster], n_boot=n_boot, seed=seed)
        pos = sub[sub[label] == 1]
        rows.append({
            group: g, **gap,
            "p_value": sign_flip_test(sub[s_female], sub[s_male], sub[cluster], n_perm=n_perm, seed=seed),
            "flip_rate": flip_rate(sub[s_female], sub[s_male], threshold),
            "tpr_gap": float((pos[s_female] >= threshold).mean() - (pos[s_male] >= threshold).mean()),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out


# ---------------------------------------------------------------- distillation
def fidelity(teacher_scores, student_scores):
    t, s = np.asarray(teacher_scores, float), np.asarray(student_scores, float)
    return {"pearson": float(stats.pearsonr(t, s)[0]),
            "spearman": float(stats.spearmanr(t, s)[0]),
            "mae": float(np.mean(np.abs(t - s)))}
