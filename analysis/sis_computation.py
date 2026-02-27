# analysis/sis_computation.py
"""
Safety Interaction Score (SIS) — per-dimension, per-pair measure.

SIS(dimension_d, pair_AB) = Score_d(Tier2_AB) - Score_d(Tier1_A)

SIS < 0: dimension d's safety degrades under composition
SIS > 0: dimension d's safety improves under composition
SIS ~ 0: dimension d is robust to composition
"""
import json, os
from collections import defaultdict
from analysis.utils import (
    load_eval_results, get_tier1_baselines, get_tier2_scores_by_pair, get_dimension_score
)


def compute_all_sis(log_dir: str = "logs") -> dict:
    """Compute SIS for every (dimension, pair) combination.

    Returns:
        {pair_id: {dimension: sis_value}}
    """
    results = load_eval_results(log_dir)
    if not results:
        print("Warning: No eval results found. Run evaluation first.")
        return {}

    baselines = get_tier1_baselines(results)
    tier2_by_pair = get_tier2_scores_by_pair(results)

    sis_results = {}

    for pair_id, pair_results in tier2_by_pair.items():
        sis_results[pair_id] = {}

        # Identify source benchmarks for this pair
        sources = pair_results[0].get("composition_sources", [])
        bench_a = sources[0] if sources else ""

        for dim in pair_results[0].get("safety_dimensions", []):
            # Tier 2 average score on this dimension
            t2_scores = []
            for r in pair_results:
                s = get_dimension_score(r, dim)
                if s is not None:
                    t2_scores.append(s)

            t2_mean = sum(t2_scores) / len(t2_scores) if t2_scores else 0

            # Tier 1 baseline for benchmark A on this dimension
            t1_baseline = baselines.get(bench_a, {}).get(dim, t2_mean)

            sis_results[pair_id][dim] = round(t2_mean - t1_baseline, 4)

    # Save
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/sis_results.json", "w") as f:
        json.dump(sis_results, f, indent=2)

    print(f"Computed SIS for {len(sis_results)} pairs")

    # Summary
    all_sis = [v for pair in sis_results.values() for v in pair.values()]
    if all_sis:
        print(f"  Mean SIS: {sum(all_sis)/len(all_sis):.4f}")
        print(f"  Min SIS: {min(all_sis):.4f} (worst degradation)")
        print(f"  Max SIS: {max(all_sis):.4f} (best improvement)")
        sig = sum(1 for s in all_sis if abs(s) > 0.1)
        print(f"  Significant (|SIS| > 0.1): {sig}/{len(all_sis)}")

    return sis_results
