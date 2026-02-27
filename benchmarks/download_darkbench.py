# benchmarks/download_darkbench.py
"""
DarkBench (Kran et al., ICLR 2025 Oral)
Dark pattern / manipulative behavior detection.
660 prompts across 6 categories.
HuggingFace: apart/darkbench
"""
from datasets import load_dataset
import json, os
from collections import Counter


def download_darkbench(output_dir="raw_data/darkbench"):
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("apart/darkbench", split="train")

    records = []
    for row in ds:
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        records.append({
            "id": row.get("id", f"db-{len(records)}"),
            "input": row["input"],
            "target": row.get("target", ""),
            "dark_pattern_category": metadata.get("dark_pattern", row.get("target", "unknown")),
        })

    with open(f"{output_dir}/darkbench_full.json", "w") as f:
        json.dump(records, f, indent=2)

    cats = Counter(r["dark_pattern_category"] for r in records)
    print(f"Downloaded {len(records)} DarkBench prompts")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    return records
