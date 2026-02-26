# benchmarks/download_injecagent.py
import json, os, subprocess

def download_injecagent(output_dir="raw_data/injecagent"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone repo (JSONL files)
    repo_dir = "raw_data/_injecagent_repo"
    if not os.path.exists(repo_dir):
        subprocess.run([
            "git", "clone", "https://github.com/uiuc-kang-lab/InjecAgent.git", repo_dir
        ], check=True)
    
    # User cases and attacker cases are separate files
    user_cases = []
    attacker_cases = []
    
    # Load user cases
    user_file = f"{repo_dir}/data/user_cases.jsonl"
    if os.path.exists(user_file):
        with open(user_file) as f:
            for line in f:
                if line.strip():
                    user_cases.append(json.loads(line))
    
    # Load attacker cases  
    for attack_type in ["direct_harm", "data_stealing"]:
        attacker_file = f"{repo_dir}/data/attacker_cases_{attack_type}.jsonl"
        if os.path.exists(attacker_file):
            with open(attacker_file) as f:
                for line in f:
                    if line.strip():
                        case = json.loads(line)
                        case["attack_type"] = attack_type
                        attacker_cases.append(case)
    
    # Also load pre-assembled test cases if available
    assembled = []
    for variant in ["base", "enhanced"]:
        assembled_file = f"{repo_dir}/data/test_cases_{variant}.jsonl"
        if os.path.exists(assembled_file):
            with open(assembled_file) as f:
                for line in f:
                    if line.strip():
                        case = json.loads(line)
                        case["variant"] = variant
                        assembled.append(case)
    
    with open(f"{output_dir}/user_cases.json", "w") as f:
        json.dump(user_cases, f, indent=2)
    with open(f"{output_dir}/attacker_cases.json", "w") as f:
        json.dump(attacker_cases, f, indent=2)
    with open(f"{output_dir}/assembled_cases.json", "w") as f:
        json.dump(assembled, f, indent=2)
    
    print(f"Downloaded {len(user_cases)} user cases, {len(attacker_cases)} attacker cases")
    print(f"Pre-assembled: {len(assembled)} test cases")
    return user_cases, attacker_cases, assembled