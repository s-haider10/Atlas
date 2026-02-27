# analysis/stats_utils.py
"""
Shared statistical helpers for ATLAS analysis modules.
Provides t-tests, confidence intervals, effect sizes, and multiple comparison correction.
"""
import math
from scipy import stats


def compute_stats(values: list) -> dict:
    """Compute descriptive statistics with 95% CI.

    Returns dict with: n, mean, std, se, ci_95.
    Returns null-safe dict if n < 2.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "se": None, "ci_95": None}
    if n == 1:
        return {"n": 1, "mean": round(values[0], 4), "std": None, "se": None, "ci_95": None}

    mean_val = sum(values) / n
    std_val = (sum((x - mean_val) ** 2 for x in values) / (n - 1)) ** 0.5
    se = std_val / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci_lo = mean_val - t_crit * se
    ci_hi = mean_val + t_crit * se

    return {
        "n": n,
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "se": round(se, 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def one_sample_ttest(values: list, popmean: float = 0) -> dict:
    """One-sample t-test: H0: mean(values) = popmean.

    Returns dict with: t_stat, p_value, cohens_d, significant.
    """
    n = len(values)
    if n < 2:
        return {"t_stat": None, "p_value": None, "cohens_d": None, "significant": False}

    t_stat, p_value = stats.ttest_1samp(values, popmean)
    mean_val = sum(values) / n
    std_val = (sum((x - mean_val) ** 2 for x in values) / (n - 1)) ** 0.5
    cohens_d = (mean_val - popmean) / std_val if std_val > 0 else 0

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "cohens_d": round(float(cohens_d), 4),
        "significant": bool(p_value < 0.05),
    }


def two_sample_ttest(values_a: list, values_b: list) -> dict:
    """Welch's t-test (unequal variance): H0: mean(a) = mean(b).

    Returns dict with: t_stat, p_value, cohens_d, significant.
    """
    if len(values_a) < 2 or len(values_b) < 2:
        return {"t_stat": None, "p_value": None, "cohens_d": None, "significant": False}

    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    var_a = sum((x - mean_a) ** 2 for x in values_a) / (len(values_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in values_b) / (len(values_b) - 1)
    pooled_std = ((var_a + var_b) / 2) ** 0.5
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "cohens_d": round(float(cohens_d), 4),
        "significant": bool(p_value < 0.05),
    }


def proportion_test(successes_a: int, n_a: int, successes_b: int, n_b: int) -> dict:
    """Two-proportion z-test (or Fisher's exact for small samples).

    H0: P(success|A) = P(success|B).
    """
    if n_a == 0 or n_b == 0:
        return {"p_value": None, "significant": False, "test": None}

    # Use Fisher's exact test for small samples
    if min(n_a, n_b) < 30:
        table = [[successes_a, n_a - successes_a],
                 [successes_b, n_b - successes_b]]
        _, p_value = stats.fisher_exact(table)
        test_name = "fisher_exact"
    else:
        table = [[successes_a, n_a - successes_a],
                 [successes_b, n_b - successes_b]]
        chi2, p_value, _, _ = stats.chi2_contingency(table, correction=True)
        test_name = "chi2"

    return {
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "test": test_name,
    }


def apply_bonferroni(p_values: list, alpha: float = 0.05) -> list:
    """Apply Bonferroni correction to a list of p-values.

    Returns list of dicts with: original_p, corrected_p, significant.
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []

    corrected_alpha = alpha / n_tests
    results = []
    for p in p_values:
        if p is None:
            results.append({"original_p": None, "corrected_p": None, "significant": False})
        else:
            corrected_p = min(p * n_tests, 1.0)
            results.append({
                "original_p": round(p, 6),
                "corrected_p": round(corrected_p, 6),
                "significant": bool(p < corrected_alpha),
            })
    return results
