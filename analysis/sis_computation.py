# analysis/sis_computation.py
"""
Safety Interaction Score (SIS) — per-dimension, per-pair measure.

SIS(dimension_d, pair_AB) = Score_d(Tier2_AB) - Score_d(Tier1_baseline)

Where the baseline is looked up via dimension_benchmark_map (explicit
dimension→benchmark mapping set at composition time), with index-based
fallback for backwards compatibility.

SIS < 0: dimension d's safety degrades under composition
SIS > 0: dimension d's safety improves under composition
SIS ~ 0: dimension d is robust to composition

Statistical test: one-sample t-test, H0: SIS = 0 (no degradation).
P-values are Bonferroni-corrected for multiple comparisons.
"""
import json, os
from collections import defaultdict
from analysis.utils import (
    load_eval_results, get_tier1_baselines, get_tier2_scores_by_pair, get_dimension_score
)
from analysis.stats_utils import compute_stats, one_sample_ttest, apply_bonferroni


def compute_all_sis(log_dir: str = "logs") -> dict:
    """Compute SIS for every (dimension, pair) combination.

    Returns:
        {pair_id: {dimension: {sis, n, p_value, ci_95, cohens_d, significant, bench_baseline}}}
    """
    results = load_eval_results(log_dir)
    if not results:
        print("Warning: No eval results found. Run evaluation first.")
        return {}

    baselines = get_tier1_baselines(results)
    tier2_by_pair = get_tier2_scores_by_pair(results)

    sis_results = {}
    all_p_values = []

    for pair_id, pair_results in tier2_by_pair.items():
        sis_results[pair_id] = {}

        # Identify source benchmarks for this pair
        sources = pair_results[0].get("composition_sources", [])
        dims = pair_results[0].get("safety_dimensions", [])
        dim_bench_map = pair_results[0].get("dimension_benchmark_map", {})

        for dim_idx, dim in enumerate(dims):
            # Use explicit dimension→benchmark map (preferred), fall back to index
            bench = dim_bench_map.get(dim)
            if bench is None:
                if dim_idx < len(sources):
                    bench = sources[dim_idx]
                else:
                    bench = sources[0] if sources else ""

            # Collect Tier 2 scores on this dimension (exclude None)
            t2_scores = []
            for r in pair_results:
                s = get_dimension_score(r, dim)
                if s is not None:
                    t2_scores.append(s)

            if not t2_scores:
                sis_results[pair_id][dim] = {
                    "sis": None, "n": 0, "reason": "no_tier2_scores",
                    "bench_baseline": bench,
                }
                continue

            # Get Tier 1 baseline for the correct benchmark on this dimension
            baseline = baselines.get(bench, {}).get(dim)
            if baseline is None:
                sis_results[pair_id][dim] = {
                    "sis": None, "n": len(t2_scores), "reason": "no_tier1_baseline",
                    "bench_baseline": bench,
                }
                continue

            # Compute per-sample SIS values (each t2 score minus baseline)
            sis_values = [s - baseline for s in t2_scores]
            stats = compute_stats(sis_values)
            ttest = one_sample_ttest(sis_values, popmean=0)

            sis_results[pair_id][dim] = {
                "sis": stats["mean"],
                "n": stats["n"],
                "std": stats["std"],
                "ci_95": stats["ci_95"],
                "t_stat": ttest["t_stat"],
                "p_value": ttest["p_value"],
                "cohens_d": ttest["cohens_d"],
                "significant": ttest["significant"],  # Will be updated after Bonferroni
                "bench_baseline": bench,
                "baseline_value": round(baseline, 4),
            }
            if ttest["p_value"] is not None:
                all_p_values.append((pair_id, dim, ttest["p_value"]))

    # Apply Bonferroni correction
    if all_p_values:
        n_tests = len(all_p_values)
        corrected_alpha = 0.05 / n_tests
        for pair_id, dim, p in all_p_values:
            sis_results[pair_id][dim]["significant_bonferroni"] = p < corrected_alpha
            sis_results[pair_id][dim]["bonferroni_alpha"] = round(corrected_alpha, 6)
            sis_results[pair_id][dim]["n_tests"] = n_tests

    # Save
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/sis_results.json", "w") as f:
        json.dump(sis_results, f, indent=2)

    # Summary
    all_sis = [v["sis"] for pair in sis_results.values() for v in pair.values()
               if v.get("sis") is not None]
    null_count = sum(1 for pair in sis_results.values() for v in pair.values()
                     if v.get("sis") is None)

    print(f"Computed SIS for {len(sis_results)} pairs")
    if all_sis:
        print(f"  Mean SIS: {sum(all_sis)/len(all_sis):.4f} (n={len(all_sis)} dimension-pair combos)")
        print(f"  Min SIS: {min(all_sis):.4f} (worst degradation)")
        sig_count = sum(1 for pair in sis_results.values() for v in pair.values()
                        if v.get("significant_bonferroni"))
        print(f"  Significant (Bonferroni p<0.05): {sig_count}/{len(all_sis)}")
    if null_count:
        print(f"  Insufficient data: {null_count} dimension-pair combos")

    return sis_results
