import itertools
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score

def _as_vector(y):
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return y.ravel()

def prediction_redundancy(y_a, y_b, method="pearson"):
    a = _as_vector(y_a)
    b = _as_vector(y_b)

    if len(np.unique(a)) <= 1 or len(np.unique(b)) <= 1:
        return 0.0

    if method == "pearson":
        r, _ = pearsonr(a, b)
        return 0.0 if np.isnan(r) else float(abs(r))
    if method == "spearman":
        r, _ = spearmanr(a, b)
        return 0.0 if np.isnan(r) else float(abs(r))
    if method == "agreement":
        return float(np.mean(a == b))
    if method == "mutual_info":
        mi = mutual_info_score(a, b)
        ha = mutual_info_score(a, a)
        hb = mutual_info_score(b, b)
        return float(mi / max(ha, hb, 1e-12))

    raise ValueError("method must be pearson, spearman, agreement, or mutual_info")

def mean_pairwise_redundancy(pred_dict, method="pearson"):
    keys = list(pred_dict.keys())
    if len(keys) < 2:
        return 0.0
    vals = []
    for a, b in itertools.combinations(keys, 2):
        vals.append(prediction_redundancy(pred_dict[a], pred_dict[b], method=method))
    return float(np.mean(vals)) if vals else 0.0
