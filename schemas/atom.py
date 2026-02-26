# schemas/atom.py
"""
Every scenario from every benchmark becomes an Atom.
Atoms are the building blocks for composition.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ToolDefinition:
    """OpenAI function-calling format."""
    name: str
    description: str
    parameters: Dict  # JSON Schema object

@dataclass
class Atom:
    # Identity
    id: str                                    # "agentharm-fraud-003"
    source_benchmark: str                      # "AgentHarm"
    source_id: str                             # Original benchmark ID
    
    # Content
    user_prompt: str                           # What the user says to the agent
    system_prompt: Optional[str] = None        # System-level context/instructions
    tools: List[ToolDefinition] = field(default_factory=list)
    environment_state: Optional[Dict] = None   # For stateful benchmarks (τ-bench, AgentDojo)
    policy_document: Optional[str] = None      # For τ-bench policy compliance scenarios
    
    # Safety dimension
    primary_dimension: str = ""                # "harmful_refusal", "injection_resistance", etc.
    secondary_dimensions: List[str] = field(default_factory=list)
    
    # Ground truth
    safe_behavior: str = ""                    # What the agent SHOULD do
    unsafe_behavior: str = ""                  # What the agent should NOT do
    ground_truth_type: str = "llm_judge"       # "deterministic" | "llm_judge" | "hybrid"
    deterministic_check: Optional[str] = None  # Python expression if deterministic
    
    # Ethical annotations (populated for all atoms, even non-moral benchmarks)
    machiavelli_tags: Dict[str, float] = field(default_factory=dict)  # 13 violation categories
    moral_foundation_tags: List[str] = field(default_factory=list)     # MFT labels
    
    # Composition metadata
    is_composable_as: List[str] = field(default_factory=list)  
    # Which operators this atom can participate in: ["IC_source", "TRC_base", "NO_norm", "SE_pressure", "MDI_dilemma"]
    domain: str = ""                           # "customer_service", "banking", "general", etc.