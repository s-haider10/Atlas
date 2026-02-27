# analysis/cdim_matrix.py
"""
Cross-Dimension Interference Matrix (CDIM).

CDIM(A->B) = Score_B(Tier2_AB) - Score_B(Tier1_B)

Measures how the *presence* of dimension A's challenge affects
performance on dimension B. The full CDIM is asymmetric (A->B != B->A).
"""
import json, os
from collections import defaultdict
from analysis.utils import (
    load_eval_results, get_tier1_baselines, get_tier2_scores_by_pair, get_dimension_score
)


def compute_cdim(log_dir: str = "logs") -> dict:
    """Compute the Cross-Dimension Interference Matrix.

    Returns:
        Nested dict {dim_a: {dim_b: cdim_value}} representing the
        asymmetric interference matrix.
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

        if len(dims) < 2 or len(sources) < 2:
            continue

        dim_a, dim_b = dims[0], dims[1]
        bench_a, bench_b = sources[0], sources[1]

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

    # Average interference values
    cdim = {}
    all_dims = sorted(set(list(interference.keys()) +
                          [d for dims in interference.values() for d in dims]))

    for dim_a in all_dims:
        cdim[dim_a] = {}
        for dim_b in all_dims:
            values = interference.get(dim_a, {}).get(dim_b, [])
            cdim[dim_a][dim_b] = round(sum(values) / len(values), 4) if values else 0.0

    # Save
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/cdim_matrix.json", "w") as f:
        json.dump(cdim, f, indent=2)

    # Also save as CSV for easy viewing
    try:
        import pandas as pd
        df = pd.DataFrame(cdim).fillna(0)
        df.to_csv("analysis/cdim_matrix.csv")
        print(f"CDIM matrix saved ({len(all_dims)}x{len(all_dims)} dimensions)")
    except ImportError:
        print(f"CDIM matrix saved as JSON ({len(all_dims)} dimensions)")

    return cdim
