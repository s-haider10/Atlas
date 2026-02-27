# composition/build_tier2.py
"""
Generate Tier 2 scenarios by applying operators to benchmark pairs.
This is the main composition pipeline.

4 benchmarks x 5 operators = up to 12 meaningful pairwise combinations.
Each pair generates 35 composed scenarios via GPT-4o.
Target: ~420 Tier 2 scenarios (n=35 per pair for statistical power).

Operator roles:
  IC  - Instruction Conflict: merge benign + adversarial instructions
  TRC - Tool Response Contamination: inject payload into tool responses
  NO  - Norm Overlay: layer ethical norm onto a task
  SE  - Sycophancy Escalation: add social pressure to refusal scenarios
  MDI - Moral Dilemma Injection: embed ethical dilemma into tool-use task
"""
import json, os, random
from tqdm import tqdm
from typing import Dict, List
from schemas.atom import Atom
from composition.operators import OPERATORS
from tools.tool_registry import generate_mock_responses

# Pairwise composition matrix for 4 benchmarks
# Each pair has a natural operator based on the benchmarks' composition roles
TIER2_PAIRS = [
    # IC: Instruction Conflict
    {"id": 1,  "bench_a": "AgentHarm",     "bench_b": "MoralStories",  "operator": "IC",  "n": 35,
     "desc": "Harmful request embedded alongside moral reasoning task"},
    {"id": 2,  "bench_a": "AgentHarm",     "bench_b": "DarkBench",     "operator": "IC",  "n": 35,
     "desc": "Harmful instruction mixed with manipulative framing"},

    # TRC: Tool Response Contamination
    {"id": 3,  "bench_a": "AgentHarm",     "bench_b": "InjecAgent",    "operator": "TRC", "n": 35,
     "desc": "Injection payload in tool responses during harmful task context"},
    {"id": 4,  "bench_a": "InjecAgent",    "bench_b": "DarkBench",     "operator": "TRC", "n": 35,
     "desc": "Dark pattern injections in tool responses"},

    # NO: Norm Overlay
    {"id": 5,  "bench_a": "AgentHarm",     "bench_b": "MoralStories",  "operator": "NO",  "n": 35,
     "desc": "Ethical constraint on harmful task — does norm help refusal?"},
    {"id": 6,  "bench_a": "InjecAgent",    "bench_b": "MoralStories",  "operator": "NO",  "n": 35,
     "desc": "Norm awareness during injection resistance — protective effect?"},

    # SE: Sycophancy Escalation
    {"id": 7,  "bench_a": "AgentHarm",     "bench_b": "DarkBench",     "operator": "SE",  "n": 35,
     "desc": "Social pressure compromises harmful task refusal"},
    {"id": 8,  "bench_a": "MoralStories",  "bench_b": "DarkBench",     "operator": "SE",  "n": 35,
     "desc": "Manipulative pressure on moral reasoning"},

    # MDI: Moral Dilemma Injection
    {"id": 9,  "bench_a": "MoralStories",  "bench_b": "AgentHarm",     "operator": "MDI", "n": 35,
     "desc": "Ethical dilemma embedded in harmful task context"},
    {"id": 10, "bench_a": "MoralStories",  "bench_b": "InjecAgent",    "operator": "MDI", "n": 35,
     "desc": "Moral dilemma distracts from injection detection"},

    # Cross-cutting
    {"id": 11, "bench_a": "DarkBench",     "bench_b": "MoralStories",  "operator": "NO",  "n": 35,
     "desc": "Norm overlay on dark pattern scenarios"},
    {"id": 12, "bench_a": "DarkBench",     "bench_b": "InjecAgent",    "operator": "IC",  "n": 35,
     "desc": "Manipulation + injection combined instruction"},
]


def select_atom_pair(atoms_a: List[Atom], atoms_b: List[Atom],
                     operator: str, used_pairs: set) -> tuple:
    """Select a compatible (atom_a, atom_b) pair not already used."""
    for _ in range(100):
        a = random.choice(atoms_a)
        b = random.choice(atoms_b)
        pair_key = f"{a.id}_{b.id}_{operator}"
        if pair_key not in used_pairs:
            used_pairs.add(pair_key)
            return a, b
    a, b = random.choice(atoms_a), random.choice(atoms_b)
    used_pairs.add(f"{a.id}_{b.id}_{operator}")
    return a, b


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
            print(f"Warning: Missing atoms for pair {pair_id} ({bench_a} x {bench_b})")
            continue

        compose_fn = OPERATORS[operator]

        for i in range(n):
            atom_a, atom_b = select_atom_pair(atoms_a, atoms_b, operator, used_pairs)

            try:
                composed = compose_fn(atom_a, atom_b)

                # Normalize tools: ensure every tool is a full dict with mock_response
                raw_tools = composed.get("tools", composed.get("tools_needed", []))
                scenario_context = {
                    "user_prompt": composed.get("composed_user_prompt", ""),
                    "safe_behavior": composed.get("safe_behavior", ""),
                    "unsafe_behavior": composed.get("unsafe_behavior", ""),
                    "operator": operator,
                }
                # TRC tools already have injected_response from the template;
                # non-TRC tools get contextual mock_response via LLM or heuristic
                enriched_tools = generate_mock_responses(raw_tools, scenario_context)

                tier2.append({
                    "id": f"atlas-t2-p{pair_id:02d}-{i:02d}",
                    "tier": 2,
                    "pair_id": pair_id,
                    "operator": operator,
                    "source_a": {"benchmark": bench_a, "atom_id": atom_a.id},
                    "source_b": {"benchmark": bench_b, "atom_id": atom_b.id},
                    "user_prompt": composed.get("composed_user_prompt", ""),
                    "system_prompt": composed.get("composed_system_prompt"),
                    "tools": enriched_tools,
                    "safety_dimensions": [
                        composed.get("dimension_a_name", atom_a.primary_dimension),
                        composed.get("dimension_b_name", atom_b.primary_dimension),
                    ],
                    "dimension_benchmark_map": {
                        composed.get("dimension_a_name", atom_a.primary_dimension): bench_a,
                        composed.get("dimension_b_name", atom_b.primary_dimension): bench_b,
                    },
                    "dimension_a_eval": composed.get("dimension_a_eval", ""),
                    "dimension_b_eval": composed.get("dimension_b_eval", ""),
                    "safe_behavior": composed.get("safe_behavior", ""),
                    "unsafe_behavior": composed.get("unsafe_behavior", ""),
                    "composition_method": operator,
                    "composition_sources": [bench_a, bench_b],
                    "raw_composed": composed,
                })

            except Exception as e:
                failed.append({"pair_id": pair_id, "index": i, "error": str(e)})
                print(f"  Failed pair {pair_id} index {i}: {e}")

    # Save
    os.makedirs("scenarios/tier2", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    with open("scenarios/tier2/tier2_scenarios.json", "w") as f:
        json.dump(tier2, f, indent=2)
    with open("logs/tier2_failures.json", "w") as f:
        json.dump(failed, f, indent=2)

    print(f"\nTier 2 complete: {len(tier2)} scenarios generated, {len(failed)} failures")
    return tier2
