# benchmarks/download_darkbench.py
from datasets import load_dataset
import json, os

def download_darkbench(output_dir="raw_data/darkbench"):
    os.makedirs(output_dir, exist_ok=True)
    
    # HuggingFace dataset
    ds = load_dataset("apart/darkbench", split="train")
    
    records = []
    for row in ds:
        records.append({
            "id": row.get("id", f"db-{len(records)}"),
            "input": row["input"],
            "target": row.get("target", ""),
            "dark_pattern_category": row.get("target", 
                row.get("metadata", {}).get("dark_pattern", "unknown")),
        })
    
    with open(f"{output_dir}/darkbench_full.json", "w") as f:
        json.dump(records, f, indent=2)
    
    # Stats by category
    from collections import Counter
    cats = Counter(r["dark_pattern_category"] for r in records)
    print(f"Downloaded {len(records)} DarkBench prompts")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    
    return records