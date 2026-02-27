# composition/build_tier1.py
"""
Sample atoms from each benchmark for Tier 1 baselines.
No composition — these are direct imports with standardized format.

4 benchmarks: AgentHarm, InjecAgent, MoralStories, DarkBench
Target: ~140 atomic scenarios (35 per benchmark) for statistical power.
"""
import json, os, random
from typing import Dict, List
from schemas.atom import Atom
from tools.tool_registry import generate_mock_responses

# Sampling budget per benchmark
TIER1_BUDGET = {
    "AgentHarm": 35,       # Stratify by harm category (8 cats)
    "InjecAgent": 35,      # Stratify by attack_type (direct_harm / data_stealing)
    "MoralStories": 35,    # Diverse norms
    "DarkBench": 35,       # ~6 per category (6 cats)
}


def stratified_sample(atoms: List[Atom], n: int, stratify_by: str) -> List[Atom]:
    """Sample n atoms, stratified by a metadata field."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for a in atoms:
        key = getattr(a, stratify_by, "unknown") if hasattr(a, stratify_by) else a.domain
        # Handle list-type fields (e.g., secondary_dimensions)
        if isinstance(key, list):
            key = key[0] if key else "unknown"
        buckets[str(key)].append(a)

    per_bucket = max(1, n // len(buckets)) if buckets else n
    sampled = []

    for key, bucket_atoms in buckets.items():
        k = min(per_bucket, len(bucket_atoms))
        sampled.extend(random.sample(bucket_atoms, k))

    # If under budget, fill from largest buckets
    while len(sampled) < n and any(buckets.values()):
        largest_key = max(buckets, key=lambda k: len(buckets[k]))
        remaining = [a for a in buckets[largest_key] if a not in sampled]
        if remaining:
            sampled.append(random.choice(remaining))
        else:
            break

    return sampled[:n]


def build_tier1(all_atoms: Dict[str, List[Atom]], seed: int = 42) -> List[dict]:
    """Build Tier 1 dataset by sampling from each benchmark."""
    random.seed(seed)

    tier1 = []
    for benchmark, budget in TIER1_BUDGET.items():
        atoms = all_atoms.get(benchmark, [])
        if not atoms:
            print(f"Warning: No atoms for {benchmark}")
            continue

        # Stratification field varies by benchmark
        stratify_field = {
            "AgentHarm": "domain",                   # harm category
            "InjecAgent": "domain",                  # attack_type
            "DarkBench": "secondary_dimensions",     # dark pattern type
            "MoralStories": "domain",                # all "ethics"
        }.get(benchmark, "domain")

        sampled = stratified_sample(atoms, budget, stratify_field)

        for atom in sampled:
            # Serialize tool definitions and enrich with mock responses
            raw_tools = [t.__dict__ for t in atom.tools]
            scenario_context = {
                "user_prompt": atom.user_prompt,
                "safe_behavior": atom.safe_behavior,
                "unsafe_behavior": atom.unsafe_behavior,
                "operator": "none",
            }
            enriched_tools = generate_mock_responses(raw_tools, scenario_context)

            tier1.append({
                "id": f"atlas-t1-{atom.id}",
                "tier": 1,
                "source_benchmark": atom.source_benchmark,
                "source_id": atom.source_id,
                "user_prompt": atom.user_prompt,
                "system_prompt": atom.system_prompt,
                "tools": enriched_tools,
                "safety_dimensions": [atom.primary_dimension] + atom.secondary_dimensions,
                "dimension_benchmark_map": {
                    dim: atom.source_benchmark
                    for dim in [atom.primary_dimension] + atom.secondary_dimensions
                },
                "safe_behavior": atom.safe_behavior,
                "unsafe_behavior": atom.unsafe_behavior,
                "ground_truth_type": atom.ground_truth_type,
                "composition_method": "none",
                "composition_sources": [atom.source_benchmark],
            })

    # Save
    os.makedirs("scenarios/tier1", exist_ok=True)
    with open("scenarios/tier1/tier1_scenarios.json", "w") as f:
        json.dump(tier1, f, indent=2)

    print(f"Built Tier 1: {len(tier1)} scenarios from {len(TIER1_BUDGET)} benchmarks")
    return tier1
