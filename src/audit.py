"""Shared audit helpers for Phase 3, built on src/metrics.py (which stays unchanged).

Conventions: scores in log-odds use threshold 0; P(Yes) uses threshold 0.5.
Gap = s_female - s_male. Bio-bootstrap CIs resample bios; name-robust CIs resample bios and first names.
"""
import numpy as np
import pandas as pd
from collections import Counter
from src import metrics

WIDE_KEYS = ["item_id", "bio_id", "occupation", "job", "label", "orig_gender", "split", "name_origin"]


def to_wide(rows, value_col, pair_col="version", pair=("female", "male"), keys=WIDE_KEYS):
    """One row per (item, pair_col value) -> one row per item with columns s_female (= pair[0])
    and s_male (= pair[1]). For a placebo, call with pair_col="arm", pair=("placebo", "orig")."""
    w = rows.pivot(index=keys, columns=pair_col, values=value_col).reset_index()
    w.columns.name = None
    w = w.rename(columns={pair[0]: "s_female", pair[1]: "s_male"})
    assert w[["s_female", "s_male"]].notna().all().all(), "missing score in a pair"
    w["all"] = "all"
    return w


def run_audit(wide, group, threshold, scale, by, n_boot, n_perm, seed, min_bios=10):
    """metrics.audit_by_group with scale/by labels and a ci_halfwidth column.
    Groups with fewer than `min_bios` distinct bios are dropped (too small for a bootstrap)."""
    keep = wide.groupby(group).bio_id.transform("nunique") >= min_bios
    w = wide[keep]
    if w.empty:
        return pd.DataFrame(columns=["scale", "by", "group", "gap", "ci_low", "ci_high", "sd", "n_clusters",
                                     "p_value", "flip_rate", "tpr_gap", "p_holm", "ci_halfwidth"])
    out = metrics.audit_by_group(w, group=group, threshold=threshold, n_boot=n_boot,
                                 n_perm=n_perm, seed=seed).rename(columns={group: "group"})
    out.insert(0, "by", by)
    out.insert(0, "scale", scale)
    out["ci_halfwidth"] = (out.ci_high - out.ci_low) / 2
    return out


def boundary_item_ids(wide_teacher_logodds, cutoff=5.0):
    """item_ids whose mean TEACHER log-odds over the two versions has |.| < cutoff."""
    m = (wide_teacher_logodds.s_female + wide_teacher_logodds.s_male) / 2
    return set(wide_teacher_logodds.loc[m.abs() < cutoff, "item_id"])


def three_scale_table(wide_lo, wide_p, group, boundary_ids, n_boot, n_perm, seed):
    """One row per group with the three things always reported together:
    log-odds gap (+CI, Holm p, flip rate, TPR gap), P(Yes) gap (+CI), boundary gap (+CI)."""
    a = run_audit(wide_lo, group, 0.0, "logodds", group, n_boot, n_perm, seed)
    p = run_audit(wide_p, group, 0.5, "p_yes", group, n_boot, n_perm, seed)
    b = run_audit(wide_lo[wide_lo.item_id.isin(boundary_ids)], group, 0.0, "boundary", group, n_boot, n_perm, seed)
    t = a[["group", "gap", "ci_low", "ci_high", "p_value", "p_holm", "flip_rate", "tpr_gap", "n_clusters"]].rename(
        columns={"gap": "gap_logodds", "ci_low": "ci_low_logodds", "ci_high": "ci_high_logodds",
                 "p_value": "p_logodds", "p_holm": "p_holm_logodds"})
    t = t.merge(p[["group", "gap", "ci_low", "ci_high"]].rename(
        columns={"gap": "gap_pyes", "ci_low": "ci_low_pyes", "ci_high": "ci_high_pyes"}), on="group", how="left")
    t = t.merge(b[["group", "gap", "ci_low", "ci_high", "flip_rate"]].rename(
        columns={"gap": "gap_boundary", "ci_low": "ci_low_boundary", "ci_high": "ci_high_boundary",
                 "flip_rate": "flip_rate_boundary"}), on="group", how="left")
    inb = wide_lo.assign(_inb=wide_lo.item_id.isin(boundary_ids)).groupby(group)._inb
    t = t.merge(pd.DataFrame({"group": inb.sum().index, "n_boundary_items": inb.sum().values,
                              "share_boundary": inb.mean().values}), on="group", how="left")
    return t


def per_bio_mean(df, col, keys=("occupation", "bio_id")):
    """Mean of `col` per bio (averages a bio's items)."""
    return df.groupby(list(keys))[col].mean().reset_index()


def flip_indicator(a, b, threshold=0.0):
    """1.0 where the decision (score >= threshold) differs between a and b."""
    return ((np.asarray(a, float) >= threshold) != (np.asarray(b, float) >= threshold)).astype(float)


def crossed_bootstrap_gap(units, n_boot=2000, seed=0, ci=0.95):
    """Name-robust CI for a mean difference. Resamples bios AND first names (two crossed random factors).

    units: one row per unit with columns
      bio_id   - bio the unit belongs to (a bio may contribute several units, e.g. 2 versions x 3 placebo draws)
      d        - the unit's difference (e.g. mean over the bio's items of s_female - s_male)
      name_a, bank_a, name_b, bank_b - the two first names compared and their banks ("indian|female", ...)
    Each replicate draws multinomial weights for bios (over unique bio_ids) and, within every bank, for
    that bank's names; a unit's weight is w_bio[bio] * w_name[name_a] * w_name[name_b].
    Returns (estimate = plain mean of d, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    d = units["d"].to_numpy(float)
    bios, bio_idx = np.unique(units["bio_id"].to_numpy(), return_inverse=True)
    keys_a = (units["bank_a"] + "|" + units["name_a"]).to_numpy()
    keys_b = (units["bank_b"] + "|" + units["name_b"]).to_numpy()
    all_keys = np.unique(np.concatenate([keys_a, keys_b]))
    idx = {k: i for i, k in enumerate(all_keys)}
    banks = {}
    for k in all_keys:
        banks.setdefault(k.rsplit("|", 1)[0], []).append(idx[k])
    ia = np.array([idx[k] for k in keys_a]); ib = np.array([idx[k] for k in keys_b])
    nb = len(bios)
    boots = np.empty(n_boot)
    for r in range(n_boot):
        wb = rng.multinomial(nb, np.full(nb, 1.0 / nb)).astype(float)[bio_idx]
        wn = np.zeros(len(all_keys))
        for members in banks.values():
            m = len(members)
            wn[members] = rng.multinomial(m, np.full(m, 1.0 / m))
        w = wb * wn[ia] * wn[ib]
        boots[r] = (w * d).sum() / w.sum() if w.sum() > 0 else np.nan
    boots = boots[np.isfinite(boots)]
    a = (1 - ci) / 2
    return float(d.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def balanced_derangement(firsts, rng):
    """Random permutation of the multiset `firsts` with out[i] != firsts[i] for every i.
    Every name appears exactly as often in `out` as in `firsts` (balanced), so additive name
    effects cancel between the original and placebo arms."""
    n = len(firsts)
    assert max(Counter(firsts).values()) <= n / 2, "one name holds more than half the bank"
    out = list(firsts)
    rng.shuffle(out)
    for _ in range(20 * n):
        bad = [i for i in range(n) if out[i] == firsts[i]]
        if not bad:
            break
        i = bad[0]
        cand = [j for j in range(n) if out[j] != firsts[i] and out[i] != firsts[j]]
        j = cand[int(rng.integers(len(cand)))]
        out[i], out[j] = out[j], out[i]
    assert all(o != f for o, f in zip(out, firsts)), "derangement failed"
    assert Counter(out) == Counter(firsts), "derangement changed the name mix"
    return out


def within_item_name_effects(rows, value_col="logodds"):
    """Name effects estimated WITHIN items. rows: control rows of conditions A + placebo_1..3 (each
    (item_id, version) scored with 4 different first names). residual = score - mean of the 4 scores of
    the same (item_id, version). Effect of a name = mean residual over its rows, within its bank.
    Returns (table with bank, first_name, n_rows, effect, se;  per-bank noise-corrected SD)."""
    r = rows.assign(first_name=rows["name"].str.split(" ").str[0],
                    bank=rows["name_origin"] + "|" + rows["version"])
    r["resid"] = r[value_col] - r.groupby(["item_id", "version"])[value_col].transform("mean")
    g = r.groupby(["bank", "first_name"]).resid.agg(["mean", "std", "count"]).reset_index()
    g = g.rename(columns={"mean": "effect", "count": "n_rows"})
    g["se"] = g["std"] / np.sqrt(g["n_rows"])
    sd = g.groupby("bank").apply(
        lambda t: float(np.sqrt(max(0.0, t.effect.var(ddof=1) - (t.se ** 2).mean())))).rename("sd_corrected")
    return g[["bank", "first_name", "n_rows", "effect", "se"]], sd.reset_index()


def half_diff_ci(x, y, n_boot=2000, seed=0, ci=0.95):
    """(mean(x) - mean(y)) / 2 with a percentile bootstrap CI; x and y resampled independently.
    Used for the original-text effect (x = per-bio gaps of female-original bios, y = male-original)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    by = y[rng.integers(0, len(y), size=(n_boot, len(y)))].mean(axis=1)
    boots = (bx - by) / 2
    a = (1 - ci) / 2
    return float((x.mean() - y.mean()) / 2), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))
