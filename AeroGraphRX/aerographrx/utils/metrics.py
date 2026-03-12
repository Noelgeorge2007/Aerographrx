"""
Performance metrics and statistical testing utilities for AeroGraphRX.

This module implements ROC analysis, AUC computation, statistical hypothesis
testing, and detection performance metrics for RF signal processing.

References:
    Usha A, Noel George. "AeroGraphRX: Graph Signal Processing for
    RF Signal Detection and Flight Tracking", 2024.
"""

import numpy as np
from scipy.special import comb
from typing import Tuple, Optional


def compute_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC (Receiver Operating Characteristic) curve.

    Sweeps decision threshold over score range to compute false positive
    rate (FPR) and true positive rate (TPR).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape (n_samples,). Values in {0, 1}.
    y_scores : np.ndarray
        Predicted scores/probabilities, shape (n_samples,).
        Usually in [0, 1] but not required.

    Returns
    -------
    fpr : np.ndarray
        False positive rates, shape (n_thresholds,).
    tpr : np.ndarray
        True positive rates, shape (n_thresholds,).
    thresholds : np.ndarray
        Decision thresholds, shape (n_thresholds,).

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.6, 0.9])
    >>> fpr, tpr, thresholds = compute_roc(y_true, y_scores)
    >>> assert len(fpr) == len(tpr)
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    # Sort by scores in descending order
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]

    # Compute TP and FP rates at each threshold
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples")

    # Initialize with all negatives (threshold at +inf)
    fpr = [0.0]
    tpr = [0.0]

    fp = 0.0
    tp = 0.0

    for i, label in enumerate(y_true_sorted):
        if label:
            tp += 1.0
        else:
            fp += 1.0

        fpr.append(fp / n_neg)
        tpr.append(tp / n_pos)

    # Thresholds
    unique_scores = np.sort(np.unique(y_scores))[::-1]
    thresholds_list = list(unique_scores) + [np.min(y_scores) - 1.0]

    return np.array(fpr), np.array(tpr), np.array(thresholds_list)


def compute_auc(
    fpr: np.ndarray,
    tpr: np.ndarray
) -> float:
    """
    Compute Area Under the ROC Curve (AUC) via trapezoidal rule.

    AUC ranges from 0 to 1, where 0.5 = random classifier, 1.0 = perfect.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates, shape (n_thresholds,).
    tpr : np.ndarray
        True positive rates, shape (n_thresholds,).

    Returns
    -------
    auc : float
        Area under ROC curve.

    Examples
    --------
    >>> fpr = np.array([0, 0.5, 1.0])
    >>> tpr = np.array([0, 0.5, 1.0])
    >>> auc = compute_auc(fpr, tpr)
    >>> assert 0 <= auc <= 1
    """
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)

    # Trapezoidal rule
    auc = np.trapz(tpr, fpr)

    return float(auc)


def delong_variance(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> float:
    """
    Compute variance of AUC using DeLong's method.

    Non-parametric estimation of AUC variance from observed data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels, shape (n_samples,).
    y_scores : np.ndarray
        Predicted scores, shape (n_samples,).

    Returns
    -------
    variance : float
        Estimated variance of AUC.

    References
    --------
    DeLong et al. (1988). "Comparing the areas under two or more correlated
    receiver operating characteristic curves: a nonparametric approach."
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    pos_scores = y_scores[y_true]
    neg_scores = y_scores[~y_true]

    # Compute Mann-Whitney statistic
    # Q1: V(Y|X=1) - variance for positive class
    # Q2: V(Y|X=0) - variance for negative class
    # V(A) = Q1 / n1 + Q2 / n0

    q1_sum = 0.0
    q2_sum = 0.0

    for pos_score in pos_scores:
        q1_sum += np.sum(neg_scores > pos_score) + 0.5 * np.sum(neg_scores == pos_score)

    for neg_score in neg_scores:
        q2_sum += np.sum(pos_scores > neg_score) + 0.5 * np.sum(pos_scores == neg_score)

    auc = q1_sum / (n_pos * n_neg)

    # Variance computation (simplified DeLong)
    Q1 = q1_sum / n_pos - auc ** 2
    Q2 = q2_sum / n_neg - auc ** 2

    variance = (auc * (1.0 - auc) + (n_pos - 1) * Q1 + (n_neg - 1) * Q2) / (n_pos * n_neg)

    return float(variance)


def delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> Tuple[float, float]:
    """
    Pairwise comparison of two AUC values using DeLong's test.

    Tests null hypothesis: AUC_a = AUC_b
    Returns z-statistic and two-tailed p-value.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    scores_a : np.ndarray
        Predictions from classifier A, shape (n_samples,).
    scores_b : np.ndarray
        Predictions from classifier B, shape (n_samples,).

    Returns
    -------
    z_stat : float
        Z-statistic for hypothesis test.
    p_value : float
        Two-tailed p-value.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> scores_a = np.array([0.1, 0.3, 0.6, 0.9])
    >>> scores_b = np.array([0.2, 0.3, 0.7, 0.8])
    >>> z, p = delong_test(y_true, scores_a, scores_b)
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=bool)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    # Compute AUCs
    fpr_a, tpr_a, _ = compute_roc(y_true, scores_a)
    auc_a = compute_auc(fpr_a, tpr_a)

    fpr_b, tpr_b, _ = compute_roc(y_true, scores_b)
    auc_b = compute_auc(fpr_b, tpr_b)

    # Variances
    var_a = delong_variance(y_true, scores_a)
    var_b = delong_variance(y_true, scores_b)

    # Covariance (simplified: use average)
    cov_ab = 0.5 * (var_a + var_b)

    # Z-statistic
    se_diff = np.sqrt(var_a + var_b - 2.0 * cov_ab)
    if se_diff < 1e-15:
        z_stat = 0.0
    else:
        z_stat = (auc_a - auc_b) / se_diff

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))

    return float(z_stat), float(p_value)


def mcnemar_test(
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_true: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for comparing two binary classifiers.

    Tests null hypothesis: classifiers have same error rate on test set.

    Parameters
    ----------
    y_pred_a : np.ndarray
        Predictions from classifier A, shape (n_samples,).
    y_pred_b : np.ndarray
        Predictions from classifier B, shape (n_samples,).
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).

    Returns
    -------
    chi2_stat : float
        Chi-squared test statistic.
    p_value : float
        P-value for null hypothesis.

    Examples
    --------
    >>> y_pred_a = np.array([0, 1, 1, 0, 1])
    >>> y_pred_b = np.array([0, 1, 0, 0, 1])
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> chi2, p = mcnemar_test(y_pred_a, y_pred_b, y_true)
    """
    from scipy.stats import chi2

    y_pred_a = np.asarray(y_pred_a, dtype=bool)
    y_pred_b = np.asarray(y_pred_b, dtype=bool)
    y_true = np.asarray(y_true, dtype=bool)

    # Errors
    error_a = (y_pred_a != y_true)
    error_b = (y_pred_b != y_true)

    # Contingency table entries
    # b: A wrong, B correct
    # c: A correct, B wrong
    b = np.sum((error_a) & (~error_b))
    c = np.sum((~error_a) & (error_b))

    # Chi-squared statistic
    if b + c == 0:
        chi2_stat = 0.0
    else:
        chi2_stat = (b - c) ** 2 / (b + c)

    # P-value (1 degree of freedom)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)

    return float(chi2_stat), float(p_value)


def bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    metric: str = 'auc'
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for classification metric.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    y_scores : np.ndarray
        Predicted scores, shape (n_samples,).
    n_bootstrap : int, optional
        Number of bootstrap samples. Default: 1000
    ci : float, optional
        Confidence level (0-1). Default: 0.95
    metric : str, optional
        Metric to evaluate: 'auc'. Default: 'auc'

    Returns
    -------
    estimate : float
        Point estimate of metric.
    lower : float
        Lower confidence bound.
    upper : float
        Upper confidence bound.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1, 0, 1])
    >>> y_scores = np.array([0.1, 0.2, 0.6, 0.8, 0.3, 0.9])
    >>> est, lo, hi = bootstrap_ci(y_true, y_scores, n_bootstrap=100)
    >>> assert lo <= est <= hi
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    n_samples = len(y_true)

    # Point estimate
    if metric == 'auc':
        fpr, tpr, _ = compute_roc(y_true, y_scores)
        estimate = compute_auc(fpr, tpr)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Bootstrap
    metric_values = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        if np.sum(y_true_boot) > 0 and np.sum(~y_true_boot) > 0:
            if metric == 'auc':
                fpr_boot, tpr_boot, _ = compute_roc(y_true_boot, y_scores_boot)
                metric_values.append(compute_auc(fpr_boot, tpr_boot))

    metric_values = np.array(metric_values)

    # Percentile confidence interval
    alpha = 1.0 - ci
    lower = np.percentile(metric_values, 100.0 * alpha / 2.0)
    upper = np.percentile(metric_values, 100.0 * (1.0 - alpha / 2.0))

    return float(estimate), float(lower), float(upper)


def youden_j(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray
) -> Tuple[float, int]:
    """
    Compute optimal threshold via Youden's J statistic.

    J = TPR - FPR, ranges from -1 to 1. Higher is better.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates, shape (n_thresholds,).
    tpr : np.ndarray
        True positive rates, shape (n_thresholds,).
    thresholds : np.ndarray
        Decision thresholds, shape (n_thresholds,).

    Returns
    -------
    optimal_threshold : float
        Threshold that maximizes J.
    best_index : int
        Index of optimal threshold.

    Examples
    --------
    >>> fpr = np.array([0, 0.2, 0.5, 1.0])
    >>> tpr = np.array([0, 0.6, 0.8, 1.0])
    >>> thresh = np.array([1.0, 0.7, 0.5, 0.0])
    >>> opt_thresh, idx = youden_j(fpr, tpr, thresh)
    """
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    # Youden J
    j_values = tpr - fpr

    # Find maximum
    best_index = np.argmax(j_values)
    optimal_threshold = thresholds[best_index]

    return float(optimal_threshold), int(best_index)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual frequencies.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    y_prob : np.ndarray
        Predicted probabilities, shape (n_samples,).
    n_bins : int, optional
        Number of bins for calibration. Default: 10

    Returns
    -------
    ece : float
        Expected calibration error, ranges [0, 1].

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    >>> ece = expected_calibration_error(y_true, y_prob)
    >>> assert 0 <= ece <= 1
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Bin probabilities
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_true)

    return float(ece)


def cohens_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute Cohen's kappa agreement statistic with 95% confidence interval.

    Kappa adjusts for chance agreement. Range: [-1, 1]
    kappa < 0.4: poor, 0.4-0.75: fair-good, > 0.75: excellent

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,).

    Returns
    -------
    kappa : float
        Cohen's kappa value.
    ci : Tuple[float, float]
        95% confidence interval (lower, upper).

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 1, 1, 0, 1])
    >>> kappa, (lo, hi) = cohens_kappa(y_true, y_pred)
    >>> assert lo <= kappa <= hi
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n = len(y_true)

    # Accuracy
    po = np.mean(y_true == y_pred)

    # Expected accuracy by chance
    classes = np.unique(y_true)
    pe = 0.0
    for c in classes:
        pe += np.mean(y_true == c) * np.mean(y_pred == c)

    # Cohen's kappa
    if pe == 1.0:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    # Variance (for confidence interval)
    # Simplified formula
    variance = (pe * (1.0 - pe)) / (n * (1.0 - pe) ** 2)
    se = np.sqrt(variance)

    # 95% CI
    z = 1.96  # 95% confidence
    lower = kappa - z * se
    upper = kappa + z * se

    return float(kappa), (float(lower), float(upper))


def detection_probability(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute probability of detection (true positive rate).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,).

    Returns
    -------
    pd : float
        Probability of detection.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    n_pos = np.sum(y_true)
    if n_pos == 0:
        return 0.0

    tp = np.sum(y_true & y_pred)
    pd = tp / n_pos

    return float(pd)


def false_alarm_probability(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute probability of false alarm (false positive rate).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,).

    Returns
    -------
    pfa : float
        Probability of false alarm.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    n_neg = np.sum(~y_true)
    if n_neg == 0:
        return 0.0

    fp = np.sum((~y_true) & y_pred)
    pfa = fp / n_neg

    return float(pfa)
