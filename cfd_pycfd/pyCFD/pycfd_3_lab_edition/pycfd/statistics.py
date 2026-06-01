import numpy as np
from scipy.stats import permutation_test

def bootstrap_gain_ci(y_true, y_base, y_fusion, score_fn, n_bootstrap=500, confidence=0.95, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_base = np.asarray(y_base)
    y_fusion = np.asarray(y_fusion)
    n = len(y_true)
    gains = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gains.append(score_fn(y_true[idx], y_fusion[idx]) - score_fn(y_true[idx], y_base[idx]))

    alpha = 1 - confidence
    return (
        float(np.percentile(gains, 100 * alpha / 2)),
        float(np.percentile(gains, 100 * (1 - alpha / 2))),
    )

def paired_permutation_pvalue(y_true, y_base, y_fusion, score_fn, n_permutations=500, random_state=42):
    y_true = np.asarray(y_true)
    y_base = np.asarray(y_base)
    y_fusion = np.asarray(y_fusion)

    def statistic(a, b):
        return score_fn(y_true, b) - score_fn(y_true, a)

    result = permutation_test(
        data=(y_base, y_fusion),
        statistic=statistic,
        permutation_type="samples",
        alternative="greater",
        n_resamples=n_permutations,
        random_state=random_state,
    )
    return float(result.pvalue)
