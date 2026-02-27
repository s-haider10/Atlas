# analysis/cdim_matrix.py
"""
Cross-Dimension Interference Matrix (CDIM).

CDIM(A->B) = Score_B(Tier2_AB) - Score_B(Tier1_B)

Measures how the *presence* of dimension A's challenge affects
performance on dimension B. The full CDIM is asymmetric (A->B != B->A).

Cells with no data are null (not 0.0). Diagonal cells are marked
"not_measurable" since a dimension cannot interfere with itself
in the pairwise design.

Statistical test: one-sample t-test per cell, H0: CDIM(A->B) = 0.
"""
import json, os
from collections import defaultdict
from analysis.utils import (
    load_eval_results, get_tier1_baselines, get_tier2_scores_by_pair, get_dimension_score
)
from analysis.stats_utils import compute_stats, one_sample_ttest


def compute_cdim(log_dir: str = "logs") -> dict:
    """Compute the Cross-Dimension Interference Matrix.

    Returns:
        {
            "matrix": {dim_a: {dim_b: cdim_value_or_null}},
            "counts": {dim_a: {dim_b: n}},
            "pvalues": {dim_a: {dim_b: p_value_or_null}},
            "stats": {dim_a: {dim_b: {mean, n, ci_95, p_value, ...}}}
        }
    """
    results = load_eval_results(log_dir)
    if not results:
        print("Warning: No eval results found.")
        return {}

    baselines = get_tier1_baselines(results)
    tier2_by_pair = get_tier2_scores_by_pair(results)

    # Collect all dimension pairs and their interference values
    interference = defaultdict(lambda: defaultdict(list))

    for pair_id, pair_results in tier2_by_pair.items():
        dims = pair_results[0].get("safety_dimensions", [])
        sources = pair_results[0].get("composition_sources", [])
        dim_bench_map = pair_results[0].get("dimension_benchmark_map", {})

        if len(dims) < 2 or len(sources) < 2:
            continue

        dim_a, dim_b = dims[0], dims[1]
        # Use explicit dimension→benchmark map (preferred), fall back to index
        bench_a = dim_bench_map.get(dim_a, sources[0])
        bench_b = dim_bench_map.get(dim_b, sources[1])

        for r in pair_results:
            # How A's presence affects B's score
            score_b = get_dimension_score(r, dim_b)
            baseline_b = baselines.get(bench_b, {}).get(dim_b)
            if score_b is not None and baseline_b is not None:
                interference[dim_a][dim_b].append(score_b - baseline_b)

            # How B's presence affects A's score
            score_a = get_dimension_score(r, dim_a)
            baseline_a = baselines.get(bench_a, {}).get(dim_a)
            if score_a is not None and baseline_a is not None:
                interference[dim_b][dim_a].append(score_a - baseline_a)

    # Build matrices
    all_dims = sorted(set(list(interference.keys()) +
                          [d for dims in interference.values() for d in dims]))

    matrix = {}
    counts = {}
    pvalues = {}
    full_stats = {}

    for dim_a in all_dims:
        matrix[dim_a] = {}
        counts[dim_a] = {}
        pvalues[dim_a] = {}
        full_stats[dim_a] = {}

        for dim_b in all_dims:
            if dim_a == dim_b:
                matrix[dim_a][dim_b] = "not_measurable"
                counts[dim_a][dim_b] = 0
                pvalues[dim_a][dim_b] = None
                full_stats[dim_a][dim_b] = {"reason": "diagonal_not_measurable"}
                continue

            values = interference.get(dim_a, {}).get(dim_b, [])
            if not values:
                matrix[dim_a][dim_b] = None
                counts[dim_a][dim_b] = 0
                pvalues[dim_a][dim_b] = None
                full_stats[dim_a][dim_b] = {"reason": "no_data", "n": 0}
                continue

            stats_result = compute_stats(values)
            ttest = one_sample_ttest(values, popmean=0)

            matrix[dim_a][dim_b] = stats_result["mean"]
            counts[dim_a][dim_b] = stats_result["n"]
            pvalues[dim_a][dim_b] = ttest["p_value"]
            full_stats[dim_a][dim_b] = {
                **stats_result,
                **ttest,
            }

    result = {
        "matrix": matrix,
        "counts": counts,
        "pvalues": pvalues,
        "stats": full_stats,
    }

    # Save
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/cdim_matrix.json", "w") as f:
        json.dump(result, f, indent=2)

    # Also save as CSV for easy viewing
    try:
        import pandas as pd
        df = pd.DataFrame(matrix)
        # Replace "not_measurable" with NaN for CSV
        df = df.replace("not_measurable", float("nan"))
        df.to_csv("analysis/cdim_matrix.csv")

        counts_df = pd.DataFrame(counts)
        counts_df.to_csv("analysis/cdim_counts.csv")

        print(f"CDIM matrix saved ({len(all_dims)}x{len(all_dims)} dimensions)")
    except ImportError:
        print(f"CDIM matrix saved as JSON ({len(all_dims)} dimensions)")

    # Summary
    valid_cells = [(da, db, full_stats[da][db])
                   for da in all_dims for db in all_dims
                   if da != db and isinstance(matrix.get(da, {}).get(db), (int, float))]
    if valid_cells:
        sig = sum(1 for _, _, s in valid_cells if s.get("significant"))
        print(f"  {len(valid_cells)} measurable cells, {sig} significant (p<0.05)")

    return result
