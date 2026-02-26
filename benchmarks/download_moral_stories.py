# benchmarks/download_moral_stories.py
from datasets import load_dataset
import json, os

def download_moral_stories(output_dir="raw_data/moral_stories"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Available on HuggingFace
    ds = load_dataset("demelin/moral_stories", "full", split="train")
    
    records = []
    for row in ds:
        records.append({
            "id": f"ms-{len(records)}",
            "norm": row["norm"],
            "situation": row["situation"],
            "intention": row["intention"],
            "moral_action": row["moral_action"],
            "moral_consequence": row["moral_consequence"],
            "immoral_action": row["immoral_action"],
            "immoral_consequence": row["immoral_consequence"],
            "label": row.get("label", None),
        })
    
    with open(f"{output_dir}/moral_stories_full.json", "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"Downloaded {len(records)} Moral Stories")
    # Extract unique norms for the norm overlay operator
    norms = list(set(r["norm"] for r in records))
    with open(f"{output_dir}/unique_norms.json", "w") as f:
        json.dump(norms, f, indent=2)
    print(f"Unique norms: {len(norms)}")
    
    return records