# pipeline/to_inspect.py
"""
Convert validated ATLAS scenarios to UK AISI Inspect Sample format.
This is the bridge between dataset generation and model evaluation.
"""
import json
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai import Task, task
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool
from scorers.multi_scorer import atlas_multi_scorer

def scenario_to_sample(scenario: dict) -> Sample:
    """Convert one ATLAS scenario dict to an Inspect Sample."""
    
    # Build input messages
    messages = []
    if scenario.get("system_prompt"):
        messages.append({"role": "system", "content": scenario["system_prompt"]})
    messages.append({"role": "user", "content": scenario["user_prompt"]})
    
    return Sample(
        id=scenario["id"],
        input=messages if len(messages) > 1 else scenario["user_prompt"],
        target=scenario.get("safe_behavior", ""),
        metadata={
            "tier": scenario["tier"],
            "safety_dimensions": scenario.get("safety_dimensions", []),
            "composition_method": scenario.get("composition_method", "none"),
            "composition_sources": scenario.get("composition_sources", []),
            "operator": scenario.get("operator", "none"),
            "safe_behavior": scenario.get("safe_behavior", ""),
            "unsafe_behavior": scenario.get("unsafe_behavior", ""),
            "dimension_a_eval": scenario.get("dimension_a_eval", ""),
            "dimension_b_eval": scenario.get("dimension_b_eval", ""),
            "pair_id": scenario.get("pair_id"),
        },
    )

def load_atlas_dataset(tier: int = None) -> MemoryDataset:
    """Load ATLAS scenarios as an Inspect dataset."""
    scenarios = []
    
    for t in ([tier] if tier else [1, 2, 3]):
        path = f"scenarios/tier{t}/tier{t}_scenarios.json"
        with open(path) as f:
            scenarios.extend(json.load(f))
    
    samples = [scenario_to_sample(s) for s in scenarios]
    return MemoryDataset(samples=samples, name=f"atlas-v2-tier{tier or 'all'}")

# Define the Inspect task
@task
def atlas_eval(tier: int = None):
    """Run ATLAS evaluation."""
    dataset = load_atlas_dataset(tier)
    
    return Task(
        dataset=dataset,
        solver=[
            use_tools(),    # Enable tool use
            generate(),     # Get model response
        ],
        scorer=atlas_multi_scorer(),  # See Step 8
    )