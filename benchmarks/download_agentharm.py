# benchmarks/download_agentharm.py
"""
AgentHarm (Andriushchenko et al., ICLR 2025)
Harmful task refusal baseline. 176 harmful + 176 benign variants.
HuggingFace: ai-safety-institute/AgentHarm
Configs: 'harmful', 'harmless_benign'
Splits: 'test_public', 'validation'
"""
from datasets import load_dataset
import json, os


def download_agentharm(output_dir="raw_data/agentharm"):
    os.makedirs(output_dir, exist_ok=True)

    records = []

    for config, is_benign in [("harmful", False), ("harmless_benign", True)]:
        ds = load_dataset("ai-safety-institute/AgentHarm", config, split="test_public")

        for row in ds:
            records.append({
                "id": f"{row['id']}-{'benign' if is_benign else 'harmful'}",
                "id_original": row.get("id_original", row["id"]),
                "name": row["name"],
                "category": row["category"],
                "prompt": row["prompt"],
                "target_functions": row.get("target_functions", []),
                "grading_function": row.get("grading_function", ""),
                "is_benign": is_benign,
                "detailed_prompt": row.get("detailed_prompt", False),
                "hint_included": row.get("hint_included", False),
            })

    with open(f"{output_dir}/agentharm_full.json", "w") as f:
        json.dump(records, f, indent=2)

    harmful_count = sum(1 for r in records if not r["is_benign"])
    benign_count = sum(1 for r in records if r["is_benign"])
    categories = set(r["category"] for r in records)

    print(f"Downloaded {len(records)} AgentHarm scenarios ({harmful_count} harmful, {benign_count} benign)")
    print(f"Categories: {categories}")
    return records
