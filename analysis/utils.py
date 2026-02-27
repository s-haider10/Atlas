# analysis/utils.py
"""
Shared utilities for loading and parsing Inspect AI eval logs.
"""
import json, os
from collections import defaultdict
from typing import Dict, List, Optional


def load_eval_results(log_dir: str = "logs") -> List[dict]:
    """Load all eval results from Inspect AI log files.

    Returns a flat list of sample-level results, each with:
        id, tier, source_benchmark, pair_id, operator,
        safety_dimensions, per_dimension_scores, overall_score
    """
    try:
        from inspect_ai.log import list_eval_logs, read_eval_log
    except ImportError:
        print("Warning: inspect_ai not available, falling back to JSON logs")
        return _load_json_results(log_dir)

    results = []
    try:
        log_infos = list_eval_logs(log_dir)
    except Exception:
        return _load_json_results(log_dir)

    for log_info in log_infos:
        try:
            log = read_eval_log(log_info)
            if log.samples is None:
                continue
            for sample in log.samples:
                score_data = {}
                overall = 0
                if sample.scores:
                    for scorer_name, score_obj in sample.scores.items():
                        if hasattr(score_obj, 'metadata') and score_obj.metadata:
                            score_data = score_obj.metadata.get("per_dimension_scores", {})
                        if hasattr(score_obj, 'value'):
                            overall = score_obj.value if isinstance(score_obj.value, (int, float)) else 0

                metadata = sample.metadata or {}
                results.append({
                    "id": sample.id,
                    "tier": metadata.get("tier", 0),
                    "source_benchmark": metadata.get("composition_sources", [""])[0] if metadata.get("composition_sources") else "",
                    "pair_id": metadata.get("pair_id"),
                    "operator": metadata.get("operator", "none"),
                    "safety_dimensions": metadata.get("safety_dimensions", []),
                    "per_dimension_scores": score_data,
                    "overall_score": overall,
                    "composition_sources": metadata.get("composition_sources", []),
                    # Tier 3 specific
                    "triple_config_id": metadata.get("triple_config_id"),
                    "base_tier2_pair_id": metadata.get("base_tier2_pair_id"),
                    "operator_chain": metadata.get("operator_chain"),
                })
        except Exception as e:
            print(f"Warning: Could not read log {log_info}: {e}")

    return results


def _load_json_results(log_dir: str) -> List[dict]:
    """Fallback: load from JSON results file."""
    results_path = os.path.join(log_dir, "eval_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return []


def get_scores_by_tier(results: List[dict]) -> Dict[int, List[dict]]:
    """Group results by tier."""
    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)
    return dict(by_tier)


def get_tier1_baselines(results: List[dict]) -> Dict[str, Dict[str, float]]:
    """Index Tier 1 scores by source_benchmark and dimension.

    Returns:
        {benchmark_name: {dimension: mean_score}}
    """
    tier1 = [r for r in results if r["tier"] == 1]

    baselines = defaultdict(lambda: defaultdict(list))
    for r in tier1:
        benchmark = r["source_benchmark"]
        for dim, score_info in r.get("per_dimension_scores", {}).items():
            score_val = score_info.get("score", 0) if isinstance(score_info, dict) else score_info
            baselines[benchmark][dim].append(score_val)

    # Average
    result = {}
    for bench, dims in baselines.items():
        result[bench] = {dim: sum(scores) / len(scores) for dim, scores in dims.items()}

    return result


def get_tier2_scores_by_pair(results: List[dict]) -> Dict[int, List[dict]]:
    """Index Tier 2 results by pair_id."""
    tier2 = [r for r in results if r["tier"] == 2]
    by_pair = defaultdict(list)
    for r in tier2:
        if r.get("pair_id") is not None:
            by_pair[r["pair_id"]].append(r)
    return dict(by_pair)


def get_dimension_score(result: dict, dimension: str) -> Optional[float]:
    """Extract score for a specific dimension from a result."""
    scores = result.get("per_dimension_scores", {})
    if dimension in scores:
        s = scores[dimension]
        return s.get("score", 0) if isinstance(s, dict) else s
    return None
