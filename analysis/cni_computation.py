# analysis/cni_computation.py
"""
Composition Nonlinearity Index (CNI).

CNI = Score(Tier3_triple) - Score(Tier2_base_pair)

Measures whether adding a third safety dimension to a Tier 2 pair causes
nonlinear degradation beyond what the pairwise composition already showed.

CNI < 0: triple composition is WORSE than pairwise (compounding effect)
CNI ~ 0: third dimension adds no extra degradation (additive)
CNI > 0: triple composition is LESS bad than pairwise (protective interaction)

Statistical test: one-sample t-test, H0: CNI = 0.
"""
import json, os
from collections import defaultdict
from analysis.utils import load_eval_results, get_tier2_scores_by_pair
from analysis.stats_utils import compute_stats, one_sample_ttest


def compute_all_cni(log_dir: str = "logs") -> dict:
    """Compute CNI for all Tier 3 configurations.

    Returns:
        {config_id: {cni, n, p_value, ci_95, cohens_d, significant, base_pair_id}}
    """
    results = load_eval_results(log_dir)
    if not results:
        print("Warning: No eval results found.")
        return {}

    tier2_by_pair = get_tier2_scores_by_pair(results)
    tier3 = [r for r in results if r["tier"] == 3]

    if not tier3:
        print("Warning: No Tier 3 results found.")
        return {}

    # Compute Tier 2 means per pair
    t2_means = {}
    for pair_id, pair_results in tier2_by_pair.items():
        scores = [r["overall_score"] for r in pair_results
                  if r.get("overall_score") is not None]
        t2_means[pair_id] = sum(scores) / len(scores) if scores else None

    # Group Tier 3 by triple_config_id — require explicit metadata
    t3_by_config = defaultdict(list)
    dropped = 0
    for r in tier3:
        config_id = r.get("triple_config_id")
        if config_id is not None:
            t3_by_config[config_id].append(r)
        else:
            dropped += 1
            print(f"  Warning: Tier 3 result '{r.get('id', '?')}' missing triple_config_id — dropped")
    if dropped:
        print(f"  {dropped} Tier 3 results dropped (missing triple_config_id in metadata)")

    if not t3_by_config:
        print("Warning: Could not group Tier 3 results by config.")
        return {}

    # Load Tier 3 configs for pair_id mapping
    t3_config_map = {}
    t3_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "scenarios/tier3/tier3_scenarios.json")
    if os.path.exists(t3_config_path):
        with open(t3_config_path) as f:
            t3_scenarios = json.load(f)
        for s in t3_scenarios:
            cid = s.get("triple_config_id")
            if cid and cid not in t3_config_map:
                t3_config_map[cid] = s.get("base_tier2_pair_id", s.get("pair_id"))

    cni_results = {}

    for config_id, t3_results in t3_by_config.items():
        t3_scores = [r["overall_score"] for r in t3_results
                     if r.get("overall_score") is not None]

        if not t3_scores:
            print(f"  Warning: Config {config_id} has no valid Tier 3 scores — skipped")
            cni_results[config_id] = {
                "cni": None, "n": 0, "reason": "no_tier3_scores",
            }
            continue

        # Get the base Tier 2 pair_id — NO global fallback
        base_pair_id = t3_results[0].get("base_tier2_pair_id")
        if base_pair_id is None:
            base_pair_id = t3_config_map.get(config_id)

        if base_pair_id is None or base_pair_id not in t2_means or t2_means[base_pair_id] is None:
            print(f"  Warning: Config {config_id} has no matching Tier 2 baseline — skipped")
            cni_results[config_id] = {
                "cni": None, "n": len(t3_scores),
                "reason": "no_tier2_baseline",
                "base_pair_id": base_pair_id,
            }
            continue

        t2_baseline = t2_means[base_pair_id]

        # Compute per-sample CNI: each Tier 3 score minus Tier 2 baseline
        cni_values = [s - t2_baseline for s in t3_scores]
        stats_result = compute_stats(cni_values)
        ttest = one_sample_ttest(cni_values, popmean=0)

        cni_results[config_id] = {
            "cni": stats_result["mean"],
            "n": stats_result["n"],
            "std": stats_result["std"],
            "ci_95": stats_result["ci_95"],
            "t_stat": ttest["t_stat"],
            "p_value": ttest["p_value"],
            "cohens_d": ttest["cohens_d"],
            "significant": ttest["significant"],
            "base_pair_id": base_pair_id,
            "t2_baseline": round(t2_baseline, 4),
        }

    # Save
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/cni_results.json", "w") as f:
        json.dump(cni_results, f, indent=2)

    valid = [v for v in cni_results.values() if v.get("cni") is not None]
    null_count = sum(1 for v in cni_results.values() if v.get("cni") is None)

    print(f"Computed CNI for {len(cni_results)} triple configs ({len(valid)} valid, {null_count} insufficient data)")
    if valid:
        vals = [v["cni"] for v in valid]
        print(f"  Mean CNI: {sum(vals)/len(vals):.4f}")
        sig = sum(1 for v in valid if v.get("significant"))
        print(f"  Significant (p<0.05): {sig}/{len(valid)}")

    return cni_results
