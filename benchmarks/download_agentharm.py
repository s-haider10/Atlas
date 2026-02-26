# benchmarks/download_agentharm.py
from datasets import load_dataset
import json, os

def download_agentharm(output_dir="raw_data/agentharm"):
    os.makedirs(output_dir, exist_ok=True)
    
    # HuggingFace dataset
    ds = load_dataset("ai-safety-institute/AgentHarm", split="test")
    
    records = []
    for row in ds:
        records.append({
            "id": row["id"],
            "id_original": row.get("id_original", row["id"]),
            "name": row["name"],
            "category": row["category"],
            "prompt": row["prompt"],
            "tools": row.get("tools", []),            # OpenAI function-calling JSON
            "target_functions": row.get("target_functions", []),
            "is_benign": row.get("is_benign", False),
            "detailed": row.get("detailed", False),
            "hint_included": row.get("hint_included", False),
        })
    
    with open(f"{output_dir}/agentharm_full.json", "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"Downloaded {len(records)} AgentHarm scenarios")
    print(f"Categories: {set(r['category'] for r in records)}")
    print(f"Benign variants: {sum(1 for r in records if r['is_benign'])}")
    return records