# benchmarks/download_moral_stories.py
"""
Moral Stories (Emelin et al., EMNLP 2021)
Norm-grounded moral reasoning with contrastive action pairs.
12,000 structured narratives.
Direct JSONL download (HuggingFace load_dataset deprecated for this dataset).
"""
import json, os, requests


def download_moral_stories(output_dir="raw_data/moral_stories"):
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/moral_stories_full.json"

    # Download JSONL directly from HuggingFace (load_dataset scripts no longer supported)
    url = "https://huggingface.co/datasets/demelin/moral_stories/resolve/main/data/moral_stories_full.jsonl"

    jsonl_path = f"{output_dir}/moral_stories_full.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"  Downloading from {url}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(jsonl_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Parse JSONL
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                records.append({
                    "id": f"ms-{len(records)}",
                    "norm": row.get("norm", ""),
                    "situation": row.get("situation", ""),
                    "intention": row.get("intention", ""),
                    "moral_action": row.get("moral_action", ""),
                    "moral_consequence": row.get("moral_consequence", ""),
                    "immoral_action": row.get("immoral_action", ""),
                    "immoral_consequence": row.get("immoral_consequence", ""),
                    "label": row.get("label", None),
                })

    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)

    # Extract unique norms
    norms = list(set(r["norm"] for r in records if r["norm"]))
    with open(f"{output_dir}/unique_norms.json", "w") as f:
        json.dump(norms, f, indent=2)

    print(f"Downloaded {len(records)} Moral Stories")
    print(f"Unique norms: {len(norms)}")
    return records
