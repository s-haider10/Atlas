# benchmarks/download_agentdojo.py
"""
AgentDojo is code-defined, not a static dataset.
We extract task definitions programmatically from the Python package.
"""
import json, os
from agentdojo.default_suites import (
    get_suite as get_workspace_suite,
)
# AgentDojo organizes tasks into suites
SUITE_NAMES = ["workspace", "banking", "travel", "slack"]

def download_agentdojo(output_dir="raw_data/agentdojo"):
    os.makedirs(output_dir, exist_ok=True)
    
    all_tasks = []
    
    for suite_name in SUITE_NAMES:
        try:
            suite = get_workspace_suite(suite_name)
            
            for task_id, task in suite.user_tasks.items():
                all_tasks.append({
                    "id": f"agentdojo-{suite_name}-{task_id}",
                    "suite": suite_name,
                    "task_id": task_id,
                    "prompt": task.PROMPT,
                    "task_type": "user_task",
                    "ground_truth": str(task.ground_truth) if hasattr(task, 'ground_truth') else None,
                })
            
            for inj_id, inj_task in suite.injection_tasks.items():
                all_tasks.append({
                    "id": f"agentdojo-{suite_name}-inj-{inj_id}",
                    "suite": suite_name,
                    "task_id": inj_id,
                    "prompt": inj_task.GOAL,
                    "task_type": "injection_task",
                })
        except Exception as e:
            print(f"Warning: Could not load suite {suite_name}: {e}")
    
    # Also extract tool definitions from suites
    # AgentDojo tools are Python functions with type annotations
    # We need their OpenAI function-calling schema
    tool_schemas = []
    for suite_name in SUITE_NAMES:
        try:
            suite = get_workspace_suite(suite_name)
            for tool in suite.tools:
                tool_schemas.append({
                    "suite": suite_name,
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,  # Already JSON schema format
                })
        except:
            pass
    
    with open(f"{output_dir}/agentdojo_tasks.json", "w") as f:
        json.dump(all_tasks, f, indent=2)
    with open(f"{output_dir}/agentdojo_tools.json", "w") as f:
        json.dump(tool_schemas, f, indent=2)
    
    print(f"Extracted {len(all_tasks)} AgentDojo tasks, {len(tool_schemas)} tool definitions")
    return all_tasks

# FALLBACK: If agentdojo package isn't installable, clone and extract
def download_agentdojo_from_git(output_dir="raw_data/agentdojo"):
    """Clone repo and parse Python files directly."""
    os.makedirs(output_dir, exist_ok=True)
    os.system("git clone https://github.com/ethz-spylab/agentdojo.git raw_data/_agentdojo_repo")
    
    # Parse task files to extract PROMPT strings and GOAL strings
    # This is a fallback — prefer the package import above
    import ast, glob
    
    tasks = []
    for pyfile in glob.glob("raw_data/_agentdojo_repo/src/agentdojo/default_suites/**/*.py", recursive=True):
        with open(pyfile) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if hasattr(target, 'id') and target.id == 'PROMPT':
                                if isinstance(item.value, ast.Constant):
                                    tasks.append({
                                        "id": f"agentdojo-{node.name}",
                                        "class_name": node.name,
                                        "prompt": item.value.value,
                                        "source_file": pyfile,
                                    })
    
    with open(f"{output_dir}/agentdojo_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Extracted {len(tasks)} tasks from source")
    return tasks