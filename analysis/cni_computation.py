# analysis/cni_computation.py
"""
Composition Nonlinearity Index (CNI).

CNI = Score(Tier3_ABC) - avg(Score(Tier2_AB), Score(Tier2_BC), Score(Tier2_AC))

CNI < 0: safety degrades MORE than predicted by pairwise interactions (compounding)
CNI ~ 0: pairwise effects are sufficient to predict triple behavior (additive)
CNI > 0: triple composition is LESS bad than predicted (protective interaction)
"""
import json, os
from collections import defaultdict
from analysis.utils import load_eval_results, get_tier2_scores_by_pair


def compute_all_cni(log_dir: str = "logs") -> dict:
    """Compute CNI for all Tier 3 scenarios.

    Returns:
        {triple_config_id: {dimension: cni_value}}
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

    # Group Tier 3 by triple_config_id (from metadata)
    t3_by_config = defaultdict(list)
    for r in tier3:
        config_id = r.get("triple_config_id")
        if config_id is not None:
            t3_by_config[config_id].append(r)
        else:
            # Fallback: extract config id from scenario id (e.g. "atlas-t3-01-00" -> 1)
            sid = r.get("id", "")
            parts = sid.split("-")
            if len(parts) >= 3 and parts[1] == "t3":
                try:
                    config_id = int(parts[2])
                    t3_by_config[config_id].append(r)
                except ValueError:
                    pass

    if not t3_by_config:
        print("Warning: Could not group Tier 3 results by config.")
        return {}

    # Compute Tier 2 averages per pair
    t2_means = {}
    for pair_id, pair_results in tier2_by_pair.items():
        scores = [r["overall_score"] for r in pair_results if r.get("overall_score") is not None]
        t2_means[pair_id] = sum(scores) / len(scores) if scores else 0

    # Load Tier 3 configs to map config_id -> base tier2 pair_id
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
        t3_scores = [r["overall_score"] for r in t3_results if r.get("overall_score") is not None]
        if not t3_scores:
            continue

        t3_mean = sum(t3_scores) / len(t3_scores)

        # Get the base Tier 2 pair_id: from metadata, or from config map
        base_pair_id = t3_results[0].get("base_tier2_pair_id")
        if base_pair_id is None:
            base_pair_id = t3_config_map.get(config_id)

        if base_pair_id is not None and base_pair_id in t2_means:
            t2_baseline = t2_means[base_pair_id]
        else:
            # Fall back to overall Tier 2 average
            all_t2_scores = [r["overall_score"] for rs in tier2_by_pair.values()
                            for r in rs if r.get("overall_score") is not None]
            t2_baseline = sum(all_t2_scores) / len(all_t2_scores) if all_t2_scores else 0

        cni_results[config_id] = round(t3_mean - t2_baseline, 4)

    # Save
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/cni_results.json", "w") as f:
        json.dump(cni_results, f, indent=2)

    print(f"Computed CNI for {len(cni_results)} triple configs")
    if cni_results:
        vals = list(cni_results.values())
        print(f"  Mean CNI: {sum(vals)/len(vals):.4f}")
        sig = sum(1 for v in vals if abs(v) > 0.1)
        print(f"  Significant (|CNI| > 0.1): {sig}/{len(vals)}")

    return cni_results
