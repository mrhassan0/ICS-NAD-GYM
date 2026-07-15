"""Confidence-interval and significance-testing helpers for the multi-seed
RL-vs-RL+LLM comparison. Two distinct kinds of uncertainty are handled:

1. Within-run uncertainty: a single trained policy's F1 is computed from an
   aggregated confusion matrix over many episodes. Episode-level bootstrap
   resampling gives a CI on that F1 without assuming a parametric form.
2. Across-seed uncertainty: each seed gives one paired (no-shaping, shaping)
   F1 observation. With only a handful of seeds, a paired t-based CI plus a
   Wilcoxon signed-rank test are reported together, since Wilcoxon alone has
   very low power at small n.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def f1_from_confusion(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def bootstrap_f1_ci(episode_confusions, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """episode_confusions: list of (tp, fp, fn, tn) per episode.

    Resamples episodes with replacement, aggregates the confusion matrix each
    time, and recomputes F1 — this respects the fact that F1 is a ratio over
    pooled counts, not a simple mean of per-episode F1s.
    """
    arr = np.asarray(episode_confusions, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    point = f1_from_confusion(*arr.sum(axis=0)[:3])
    if n < 2:
        return {"point": point, "lo": point, "hi": point, "n_episodes": n, "n_boot": 0}
    boot_f1s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tp, fp, fn, _tn = arr[idx].sum(axis=0)
        boot_f1s[b] = f1_from_confusion(tp, fp, fn)
    lo, hi = np.percentile(boot_f1s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_episodes": n, "n_boot": n_boot,
            "boot_std": float(boot_f1s.std(ddof=1))}


def mean_t_ci(values, alpha: float = 0.05):
    """Standard t-distribution CI for the mean of a (small) sample."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return {"mean": mean, "lo": mean, "hi": mean, "n": n}
    sem = arr.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return {"mean": mean, "lo": mean - tcrit * sem, "hi": mean + tcrit * sem, "n": n, "sem": float(sem)}


def paired_diff_ci_and_test(baseline_values, treatment_values, alpha: float = 0.05):
    """Paired t-based CI on (treatment - baseline), plus a Wilcoxon signed-rank
    test on the same pairs. Both are reported together deliberately: the
    t-CI communicates the estimated effect size and its uncertainty, while
    Wilcoxon is a distribution-free significance check. At small n neither is
    individually definitive; the honest read is both together.
    """
    base = np.asarray(baseline_values, dtype=np.float64)
    treat = np.asarray(treatment_values, dtype=np.float64)
    assert len(base) == len(treat), "paired arrays must be the same length"
    diffs = treat - base
    n = len(diffs)
    ci = mean_t_ci(diffs, alpha=alpha)

    result = {
        "n_pairs": n,
        "diffs": diffs.tolist(),
        "mean_diff": ci["mean"],
        "ci_lo": ci["lo"],
        "ci_hi": ci["hi"],
    }
    if n >= 2 and not np.allclose(diffs, diffs[0]):
        try:
            wstat, wp = stats.wilcoxon(treat, base)
            result["wilcoxon_stat"] = float(wstat)
            result["wilcoxon_p"] = float(wp)
        except ValueError as e:
            result["wilcoxon_stat"] = None
            result["wilcoxon_p"] = None
            result["wilcoxon_note"] = str(e)
    else:
        result["wilcoxon_stat"] = None
        result["wilcoxon_p"] = None
        result["wilcoxon_note"] = "all diffs identical or n<2; signed-rank test undefined"

    # One-sample t-test on the diffs as a second, parametric significance check.
    if n >= 2:
        tstat, tp_ = stats.ttest_1samp(diffs, popmean=0.0)
        result["paired_ttest_stat"] = float(tstat)
        result["paired_ttest_p"] = float(tp_)
    else:
        result["paired_ttest_stat"] = None
        result["paired_ttest_p"] = None

    return result
