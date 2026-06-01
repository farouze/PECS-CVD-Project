import numpy as np
from .metrics import score
from .statistics import bootstrap_gain_ci, paired_permutation_pvalue
from .redundancy import prediction_redundancy, mean_pairwise_redundancy
from .report import CFDResult
from .utils import strength_from_index

def _make_result(name, domains, result_type, y_true, y_base, y_fusion, base_scores, fusion_score,
                 redundancy, task, metric, alpha, min_gain, n_bootstrap, n_permutations, random_state, metadata=None):
    def score_fn(yt, yp):
        return score(yt, yp, task=task, metric=metric)

    best_score = max(base_scores.values())
    best_domain = max(base_scores, key=base_scores.get)
    fusion_gain = fusion_score - best_score

    ci = bootstrap_gain_ci(y_true, y_base, y_fusion, score_fn, n_bootstrap=n_bootstrap, random_state=random_state)
    p_value = paired_permutation_pvalue(y_true, y_base, y_fusion, score_fn, n_permutations=n_permutations, random_state=random_state)

    cfd_score = fusion_gain * (1 - redundancy)
    cfd_index = 100 * cfd_score
    is_comp = bool(fusion_gain > min_gain and p_value < alpha and ci[0] > 0)

    scores = {k: float(v) for k, v in base_scores.items()}
    scores.update({
        "best_domain": best_domain,
        "best_domain_score": float(best_score),
        "fusion": float(fusion_score),
    })

    return CFDResult(
        name=name,
        domains=tuple(domains),
        type=result_type,
        is_complementary=is_comp,
        hypothesis_decision="Reject H0" if is_comp else "Fail to reject H0",
        fusion_gain=float(fusion_gain),
        p_value=float(p_value),
        confidence_interval=ci,
        redundancy=float(redundancy),
        cfd_score=float(cfd_score),
        cfd_index=float(cfd_index),
        strength=strength_from_index(cfd_index),
        scores=scores,
        metadata=metadata,
    )

def cfd_test_predictions(
    y_true,
    y_a,
    y_b,
    y_fusion,
    name_a="A",
    name_b="B",
    task="classification",
    metric="accuracy",
    alpha=0.05,
    min_gain=0.01,
    redundancy_method="pearson",
    n_bootstrap=500,
    n_permutations=500,
    random_state=42,
):
    y_true = np.asarray(y_true)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    y_fusion = np.asarray(y_fusion)

    score_a = score(y_true, y_a, task=task, metric=metric)
    score_b = score(y_true, y_b, task=task, metric=metric)
    fusion_score = score(y_true, y_fusion, task=task, metric=metric)
    y_base = y_a if score_a >= score_b else y_b
    redundancy = prediction_redundancy(y_a, y_b, method=redundancy_method)

    return _make_result(
        name=f"{name_a}+{name_b}",
        domains=(name_a, name_b),
        result_type="prediction_pairwise",
        y_true=y_true,
        y_base=y_base,
        y_fusion=y_fusion,
        base_scores={name_a: score_a, name_b: score_b},
        fusion_score=fusion_score,
        redundancy=redundancy,
        task=task,
        metric=metric,
        alpha=alpha,
        min_gain=min_gain,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_state=random_state,
        metadata={"redundancy_method": redundancy_method},
    )

def cfd_test_full_predictions(
    y_true,
    y_domain_preds,
    y_fusion,
    task="classification",
    metric="accuracy",
    alpha=0.05,
    min_gain=0.01,
    redundancy_method="pearson",
    n_bootstrap=500,
    n_permutations=500,
    random_state=42,
):
    y_true = np.asarray(y_true)
    y_fusion = np.asarray(y_fusion)

    if not isinstance(y_domain_preds, dict) or len(y_domain_preds) < 2:
        raise ValueError("y_domain_preds must be a dictionary with at least two prediction arrays")

    base_scores = {
        name: score(y_true, pred, task=task, metric=metric)
        for name, pred in y_domain_preds.items()
    }

    best_name = max(base_scores, key=base_scores.get)
    y_base = np.asarray(y_domain_preds[best_name])
    fusion_score = score(y_true, y_fusion, task=task, metric=metric)
    redundancy = mean_pairwise_redundancy(y_domain_preds, method=redundancy_method)

    return _make_result(
        name="+".join(y_domain_preds.keys()),
        domains=tuple(y_domain_preds.keys()),
        result_type="prediction_full_fusion",
        y_true=y_true,
        y_base=y_base,
        y_fusion=y_fusion,
        base_scores=base_scores,
        fusion_score=fusion_score,
        redundancy=redundancy,
        task=task,
        metric=metric,
        alpha=alpha,
        min_gain=min_gain,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        random_state=random_state,
        metadata={"best_unimodal": best_name, "full_redundancy": "mean_pairwise_prediction_redundancy"},
    )
