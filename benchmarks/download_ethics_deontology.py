# benchmarks/download_ethics.py
import pandas as pd
import json, os, subprocess

def download_ethics(output_dir="raw_data/ethics"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone the ETHICS dataset repo
    repo_dir = "raw_data/_ethics_repo"
    if not os.path.exists(repo_dir):
        subprocess.run([
            "git", "clone", "https://github.com/hendrycks/ethics.git", repo_dir
        ], check=True)
    
    # Deontology subset is in CSV format
    records = []
    for split in ["train", "test"]:
        csv_path = f"{repo_dir}/deontology/{split}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None, 
                           names=["label", "scenario", "excuse"])
            # Some files have group_id as first column
            # Adjust parsing based on actual column count
            for _, row in df.iterrows():
                records.append({
                    "id": f"ethics-deont-{len(records)}",
                    "label": int(row["label"]),  # 0=unreasonable, 1=reasonable
                    "scenario": str(row["scenario"]),
                    "excuse": str(row["excuse"]),
                    "split": split,
                })
    
    with open(f"{output_dir}/deontology_full.json", "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"Downloaded {len(records)} ETHICS-Deontology examples")
    print(f"Reasonable: {sum(1 for r in records if r['label']==1)}, "
          f"Unreasonable: {sum(1 for r in records if r['label']==0)}")
    return records