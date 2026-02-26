# composition/build_tier1.py
"""
Sample atoms from each benchmark for Tier 1 baselines.
No composition — these are direct imports with standardized format.
"""
import json, random
from typing import Dict, List
from schemas.atom import Atom

# Sampling budget per benchmark
TIER1_BUDGET = {
    "AgentHarm": 15,        # Stratify by 11 harm categories
    "AgentDojo": 10,        # 2 per suite
    "tau_bench": 10,        # Stratify retail/airline
    "InjecAgent": 10,       # 5 direct_harm + 5 data_stealing
    "MACHIAVELLI": 10,      # Highest multi-violation nodes
    "MoralStories": 10,     # Diverse norms
    "ETHICS-Deontology": 10,# 5 reasonable + 5 unreasonable
    "MIC": 10,              # Stratify by moral foundation
    "DarkBench": 15,        # ~2-3 per category
}

def stratified_sample(atoms: List[Atom], n: int, stratify_by: str) -> List[Atom]:
    """Sample n atoms, stratified by a metadata field."""
    from collections import defaultdict
    
    buckets = defaultdict(list)
    for a in atoms:
        key = getattr(a, stratify_by, "unknown") if hasattr(a, stratify_by) else a.domain
        buckets[key].append(a)
    
    per_bucket = max(1, n // len(buckets))
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
            "AgentHarm": "domain",          # harm category
            "DarkBench": "secondary_dimensions",  # dark pattern type
            "MIC": "moral_foundation_tags",
            "ETHICS-Deontology": "safe_behavior",  # reasonable vs unreasonable
        }.get(benchmark, "domain")
        
        sampled = stratified_sample(atoms, budget, stratify_field)
        
        for atom in sampled:
            tier1.append({
                "id": f"atlas-t1-{atom.id}",
                "tier": 1,
                "source_benchmark": atom.source_benchmark,
                "source_id": atom.source_id,
                "user_prompt": atom.user_prompt,
                "system_prompt": atom.system_prompt,
                "tools": [t.__dict__ for t in atom.tools],
                "safety_dimensions": [atom.primary_dimension] + atom.secondary_dimensions,
                "safe_behavior": atom.safe_behavior,
                "unsafe_behavior": atom.unsafe_behavior,
                "ground_truth_type": atom.ground_truth_type,
                "composition_method": "none",
                "composition_sources": [atom.source_benchmark],
            })
    
    # Save
    with open("scenarios/tier1/tier1_scenarios.json", "w") as f:
        json.dump(tier1, f, indent=2)
    
    print(f"Built Tier 1: {len(tier1)} scenarios from {len(TIER1_BUDGET)} benchmarks")
    return tier1