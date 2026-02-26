# composition/build_tier2.py
"""
Generate Tier 2 scenarios by applying operators to benchmark pairs.
This is the main composition pipeline.
"""
import json, os, random
from tqdm import tqdm
from typing import Dict, List
from schemas.atom import Atom
from composition.operators import OPERATORS

# The 15 priority pairs from the engineering spec
TIER2_PAIRS = [
    {"id": 1,  "bench_a": "tau_bench",     "bench_b": "AgentHarm",         "operator": "IC",  "n": 10},
    {"id": 2,  "bench_a": "AgentDojo",     "bench_b": "InjecAgent",        "operator": "TRC", "n": 10},
    {"id": 3,  "bench_a": "AgentHarm",     "bench_b": "MoralStories",      "operator": "NO",  "n": 10},
    {"id": 4,  "bench_a": "AgentHarm",     "bench_b": "DarkBench",         "operator": "SE",  "n": 10},
    {"id": 5,  "bench_a": "tau_bench",     "bench_b": "MACHIAVELLI",       "operator": "MDI", "n": 10},
    {"id": 6,  "bench_a": "AgentDojo",     "bench_b": "DarkBench",         "operator": "SE",  "n": 10},
    {"id": 7,  "bench_a": "tau_bench",     "bench_b": "InjecAgent",        "operator": "TRC", "n": 10},
    {"id": 8,  "bench_a": "AgentHarm",     "bench_b": "ETHICS-Deontology", "operator": "NO",  "n": 10},
    {"id": 9,  "bench_a": "tau_bench",     "bench_b": "MIC",               "operator": "NO",  "n": 10},
    {"id": 10, "bench_a": "MACHIAVELLI",   "bench_b": "DarkBench",         "operator": "SE",  "n": 10},
    {"id": 11, "bench_a": "AgentDojo",     "bench_b": "MoralStories",      "operator": "NO",  "n": 10},
    {"id": 12, "bench_a": "InjecAgent",    "bench_b": "DarkBench",         "operator": "TRC", "n": 10},
    {"id": 13, "bench_a": "AgentHarm",     "bench_b": "MIC",               "operator": "IC",  "n": 10},
    {"id": 14, "bench_a": "tau_bench",     "bench_b": "ETHICS-Deontology", "operator": "MDI", "n": 10},
    {"id": 15, "bench_a": "AgentDojo",     "bench_b": "MACHIAVELLI",       "operator": "MDI", "n": 10},
]

def select_atom_pair(atoms_a: List[Atom], atoms_b: List[Atom], 
                     operator: str, used_pairs: set) -> tuple:
    """Select a compatible (atom_a, atom_b) pair not already used."""
    for _ in range(100):  # max attempts
        a = random.choice(atoms_a)
        b = random.choice(atoms_b)
        pair_key = f"{a.id}_{b.id}_{operator}"
        if pair_key not in used_pairs:
            used_pairs.add(pair_key)
            return a, b
    # Fallback: allow repeats
    return random.choice(atoms_a), random.choice(atoms_b)

def build_tier2(all_atoms: Dict[str, List[Atom]], seed: int = 42) -> List[dict]:
    """Generate all Tier 2 composed scenarios."""
    random.seed(seed)
    
    tier2 = []
    used_pairs = set()
    failed = []
    
    for pair_config in tqdm(TIER2_PAIRS, desc="Tier 2 pairs"):
        pair_id = pair_config["id"]
        bench_a = pair_config["bench_a"]
        bench_b = pair_config["bench_b"]
        operator = pair_config["operator"]
        n = pair_config["n"]
        
        atoms_a = all_atoms.get(bench_a, [])
        atoms_b = all_atoms.get(bench_b, [])
        
        if not atoms_a or not atoms_b:
            print(f"Warning: Missing atoms for pair {pair_id} ({bench_a} × {bench_b})")
            continue
        
        compose_fn = OPERATORS[operator]
        
        for i in range(n):
            atom_a, atom_b = select_atom_pair(atoms_a, atoms_b, operator, used_pairs)
            
            try:
                composed = compose_fn(atom_a, atom_b)
                
                tier2.append({
                    "id": f"atlas-t2-p{pair_id}-{i:02d}",
                    "tier": 2,
                    "pair_id": pair_id,
                    "operator": operator,
                    "source_a": {"benchmark": bench_a, "atom_id": atom_a.id},
                    "source_b": {"benchmark": bench_b, "atom_id": atom_b.id},
                    "user_prompt": composed.get("composed_user_prompt", ""),
                    "system_prompt": composed.get("composed_system_prompt"),
                    "tools": composed.get("tools", composed.get("tools_needed", [])),
                    "safety_dimensions": [
                        composed.get("dimension_a_name", atom_a.primary_dimension),
                        composed.get("dimension_b_name", atom_b.primary_dimension),
                    ],
                    "dimension_a_eval": composed.get("dimension_a_eval", ""),
                    "dimension_b_eval": composed.get("dimension_b_eval", ""),
                    "safe_behavior": composed.get("safe_behavior", ""),
                    "unsafe_behavior": composed.get("unsafe_behavior", ""),
                    "composition_method": operator,
                    "composition_sources": [bench_a, bench_b],
                    "raw_composed": composed,  # Full GPT-4o output for debugging
                })
                
            except Exception as e:
                failed.append({"pair_id": pair_id, "index": i, "error": str(e)})
                print(f"  Failed pair {pair_id} index {i}: {e}")
    
    # Save
    os.makedirs("scenarios/tier2", exist_ok=True)
    with open("scenarios/tier2/tier2_scenarios.json", "w") as f:
        json.dump(tier2, f, indent=2)
    with open("logs/tier2_failures.json", "w") as f:
        json.dump(failed, f, indent=2)
    
    print(f"\nTier 2 complete: {len(tier2)} scenarios generated, {len(failed)} failures")
    return tier2