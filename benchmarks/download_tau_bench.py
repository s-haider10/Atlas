# benchmarks/download_tau_bench.py
import json, os, subprocess

def download_tau_bench(output_dir="raw_data/tau_bench"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone the repo
    repo_dir = "raw_data/_tau_bench_repo"
    if not os.path.exists(repo_dir):
        subprocess.run([
            "git", "clone", "https://github.com/sierra-research/tau-bench.git", repo_dir
        ], check=True)
    
    # τ-bench tasks are in JSON files within the repo
    # Structure: {domain}/tasks.json with policy docs in {domain}/wiki/
    tasks = []
    policies = {}
    
    for domain in ["retail", "airline"]:
        task_file = f"{repo_dir}/data/{domain}/tasks.json"
        if os.path.exists(task_file):
            with open(task_file) as f:
                domain_tasks = json.load(f)
            for t in domain_tasks:
                t["domain"] = domain
                t["id"] = f"tau-{domain}-{t.get('id', len(tasks))}"
                tasks.append(t)
        
        # Extract policy documents (Markdown files)
        wiki_dir = f"{repo_dir}/data/{domain}/wiki"
        if os.path.exists(wiki_dir):
            policies[domain] = {}
            for md_file in os.listdir(wiki_dir):
                if md_file.endswith(".md"):
                    with open(f"{wiki_dir}/{md_file}") as f:
                        policies[domain][md_file] = f.read()
    
    # Extract database schemas
    databases = {}
    for domain in ["retail", "airline"]:
        db_file = f"{repo_dir}/data/{domain}/database.json"
        if os.path.exists(db_file):
            with open(db_file) as f:
                databases[domain] = json.load(f)
    
    with open(f"{output_dir}/tau_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    with open(f"{output_dir}/tau_policies.json", "w") as f:
        json.dump(policies, f, indent=2)
    with open(f"{output_dir}/tau_databases.json", "w") as f:
        json.dump(databases, f, indent=2)
    
    print(f"Downloaded {len(tasks)} τ-bench tasks across {len(policies)} domains")
    print(f"Policy documents: {sum(len(v) for v in policies.values())} files")
    return tasks, policies, databases