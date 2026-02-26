# benchmarks/download_mic.py
import pandas as pd
import json, os

def download_mic(output_dir="raw_data/mic"):
    os.makedirs(output_dir, exist_ok=True)
    
    # MIC is available via Caleb Ziems' dataset page
    # Primary data file: CSV with prompt-reply-RoT triples
    # Try HuggingFace first, fallback to direct download
    
    try:
        from datasets import load_dataset
        ds = load_dataset("SALT-NLP/MIC", split="train")
        records = [dict(row) for row in ds]
    except:
        # Direct download fallback
        import subprocess
        repo_dir = "raw_data/_mic_repo"
        if not os.path.exists(repo_dir):
            subprocess.run([
                "git", "clone", "https://github.com/SALT-NLP/mic.git", repo_dir
            ], check=True)
        
        # Parse the CSV
        csv_path = f"{repo_dir}/data/mic.csv"
        df = pd.read_csv(csv_path)
        records = df.to_dict(orient="records")
    
    # Normalize field names
    normalized = []
    for r in records:
        normalized.append({
            "id": f"mic-{len(normalized)}",
            "prompt": r.get("Q", r.get("prompt", "")),
            "reply": r.get("A", r.get("reply", "")),
            "rot": r.get("rot", r.get("rule_of_thumb", "")),
            "rot_moral_foundations": r.get("rot-moral-foundations", ""),
            "rot_agree": r.get("rot-agree", None),
            "moral_foundation_care": "care" in str(r.get("rot-moral-foundations", "")).lower(),
            "moral_foundation_fairness": "fairness" in str(r.get("rot-moral-foundations", "")).lower(),
            "moral_foundation_loyalty": "loyalty" in str(r.get("rot-moral-foundations", "")).lower(),
            "moral_foundation_authority": "authority" in str(r.get("rot-moral-foundations", "")).lower(),
            "moral_foundation_purity": "purity" in str(r.get("rot-moral-foundations", "")).lower(),
            "moral_foundation_liberty": "liberty" in str(r.get("rot-moral-foundations", "")).lower(),
        })
    
    with open(f"{output_dir}/mic_full.json", "w") as f:
        json.dump(normalized, f, indent=2)
    
    print(f"Downloaded {len(normalized)} MIC examples")
    return normalized