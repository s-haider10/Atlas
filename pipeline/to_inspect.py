# pipeline/to_inspect.py
"""
Convert validated ATLAS scenarios to UK AISI Inspect Sample format.
This is the bridge between dataset generation and model evaluation.
"""
import json, sys, os

# Ensure project root is on sys.path so Inspect can find our modules
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai import Task, task
from inspect_ai.solver import solver, Solver, TaskState, Generate, generate
from scorers.multi_scorer import atlas_multi_scorer
from tools.tool_registry import build_tools_for_scenario

def scenario_to_sample(scenario: dict) -> Sample:
    """Convert one ATLAS scenario dict to an Inspect Sample."""
    
    # Build input messages
    messages = []
    if scenario.get("system_prompt"):
        messages.append({"role": "system", "content": scenario["system_prompt"]})
    messages.append({"role": "user", "content": scenario["user_prompt"]})
    
    # Tools are full dicts (with name/description/mock_response or injected_response)
    # generated at composition time. Pass them through verbatim.
    tools = scenario.get("tools", [])

    # Build tool audit manifest: what tools are available and what they return.
    # This is included in metadata so LLM judge and deterministic checker can
    # audit tool interactions without needing to re-run the tools.
    tool_audit = []
    for t in tools:
        if isinstance(t, dict):
            entry = {"name": t.get("name", "unknown"), "description": t.get("description", "")}
            if t.get("injected_response"):
                entry["response_type"] = "injected"
                entry["response"] = t["injected_response"]
            elif t.get("mock_response"):
                entry["response_type"] = "mock"
                entry["response"] = t["mock_response"]
            tool_audit.append(entry)
        elif isinstance(t, str):
            tool_audit.append({"name": t, "response_type": "heuristic"})

    return Sample(
        id=scenario["id"],
        input=messages if len(messages) > 1 else scenario["user_prompt"],
        target=scenario.get("safe_behavior", ""),
        metadata={
            "tier": scenario["tier"],
            "safety_dimensions": scenario.get("safety_dimensions", []),
            "dimension_benchmark_map": scenario.get("dimension_benchmark_map", {}),
            "composition_method": scenario.get("composition_method", "none"),
            "composition_sources": scenario.get("composition_sources", []),
            "operator": scenario.get("operator", "none"),
            "safe_behavior": scenario.get("safe_behavior", ""),
            "unsafe_behavior": scenario.get("unsafe_behavior", ""),
            "dimension_a_eval": scenario.get("dimension_a_eval", ""),
            "dimension_b_eval": scenario.get("dimension_b_eval", ""),
            "dimension_c_eval": scenario.get("dimension_c_eval", ""),
            "pair_id": scenario.get("pair_id"),
            "tools": tools,
            "tool_audit": tool_audit,
            # Tier 3 specific
            "triple_config_id": scenario.get("triple_config_id"),
            "base_tier2_id": scenario.get("base_tier2_id"),
            "base_tier2_pair_id": scenario.get("base_tier2_pair_id"),
            "operator_chain": scenario.get("operator_chain"),
        },
    )

def load_atlas_dataset(tier: int = None) -> MemoryDataset:
    """Load ATLAS scenarios as an Inspect dataset."""
    scenarios = []
    
    for t in ([tier] if tier else [1, 2, 3]):
        path = os.path.join(_project_root, f"scenarios/tier{t}/tier{t}_scenarios.json")
        with open(path) as f:
            scenarios.extend(json.load(f))
    
    samples = [scenario_to_sample(s) for s in scenarios]
    return MemoryDataset(samples=samples, name=f"atlas-v2-tier{tier or 'all'}")

@solver
def inject_scenario_tools() -> Solver:
    """Per-sample solver that injects tools from scenario metadata."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tools_data = state.metadata.get("tools", [])
        operator = state.metadata.get("operator", "none")
        if tools_data:
            state.tools = build_tools_for_scenario(tools_data, operator)
        return state
    return solve


# Define the Inspect task
@task
def atlas_eval(tier: int = None):
    """Run ATLAS evaluation."""
    dataset = load_atlas_dataset(tier)

    return Task(
        dataset=dataset,
        solver=[
            inject_scenario_tools(),  # Per-sample tool injection
            generate(),               # Get model response
        ],
        scorer=atlas_multi_scorer(),
    )