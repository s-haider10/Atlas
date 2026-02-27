# benchmarks/download_injecagent.py
"""
InjecAgent (Zhan et al., ACL 2024)
Indirect prompt injection via tool response contamination.
1,054 test cases = 17 user tools x 62 attacker tools.
GitHub: uiuc-kang-lab/InjecAgent
Data files: data/user_cases.jsonl, data/attacker_cases_dh.jsonl, data/attacker_cases_ds.jsonl
"""
import json, os, subprocess


def download_injecagent(output_dir="raw_data/injecagent"):
    os.makedirs(output_dir, exist_ok=True)

    repo_dir = "raw_data/_injecagent_repo"
    if not os.path.exists(repo_dir):
        subprocess.run([
            "git", "clone", "https://github.com/uiuc-kang-lab/InjecAgent.git", repo_dir
        ], check=True)

    user_cases = []
    attacker_cases = []

    # Load user cases
    user_file = f"{repo_dir}/data/user_cases.jsonl"
    if os.path.exists(user_file):
        with open(user_file) as f:
            for line in f:
                if line.strip():
                    user_cases.append(json.loads(line))
    else:
        print(f"  Warning: {user_file} not found")

    # Load attacker cases — files use _dh (direct harm) and _ds (data stealing)
    for suffix, attack_type in [("dh", "direct_harm"), ("ds", "data_stealing")]:
        attacker_file = f"{repo_dir}/data/attacker_cases_{suffix}.jsonl"
        if os.path.exists(attacker_file):
            with open(attacker_file) as f:
                for line in f:
                    if line.strip():
                        case = json.loads(line)
                        case["attack_type"] = attack_type
                        attacker_cases.append(case)
        else:
            print(f"  Warning: {attacker_file} not found")

    # Build assembled test cases by cross-product (user x attacker)
    assembled = []
    for uc in user_cases:
        for ac in attacker_cases:
            assembled.append({
                "id": f"injecagent-u{uc.get('User Case ID', len(assembled))}-a{ac.get('Attacker Case ID', '')}",
                "user_tool": uc.get("User Tool", ""),
                "user_instruction": uc.get("User Instruction", ""),
                "tool_response_template": uc.get("Tool Response Template", ""),
                "attacker_tool": ac.get("Attacker Tool", ""),
                "attacker_instruction": ac.get("Attacker Instruction", ""),
                "attack_type": ac.get("attack_type", "unknown"),
                "attack_category": ac.get("Attack Category", ""),
            })

    with open(f"{output_dir}/user_cases.json", "w") as f:
        json.dump(user_cases, f, indent=2)
    with open(f"{output_dir}/attacker_cases.json", "w") as f:
        json.dump(attacker_cases, f, indent=2)
    with open(f"{output_dir}/assembled_cases.json", "w") as f:
        json.dump(assembled, f, indent=2)

    print(f"Downloaded {len(user_cases)} user cases, {len(attacker_cases)} attacker cases")
    print(f"Assembled: {len(assembled)} cross-product test cases")
    return user_cases, attacker_cases, assembled
