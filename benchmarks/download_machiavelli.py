# benchmarks/download_machiavelli.py
import json, os, subprocess

def download_machiavelli(output_dir="raw_data/machiavelli"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone main repo
    repo_dir = "raw_data/_machiavelli_repo"
    if not os.path.exists(repo_dir):
        subprocess.run([
            "git", "clone", "https://github.com/aypan17/machiavelli.git", repo_dir
        ], check=True)
    
    # MACHIAVELLI stores game data as individual JSON files per game
    # Each game has scenes with annotated choices
    # The annotations include 13 ethical violation categories
    
    # Also available as a processed dataset:
    # https://github.com/wassname/machiavelli_as_ds
    subprocess.run([
        "git", "clone", "https://github.com/wassname/machiavelli_as_ds.git",
        "raw_data/_machiavelli_ds"
    ], check=True)
    
    # Extract high-ethical-cost decision nodes
    # These are the nodes we'll use for composition
    decision_nodes = []
    
    game_dir = f"{repo_dir}/game_data"
    if os.path.exists(game_dir):
        for game_file in os.listdir(game_dir):
            if game_file.endswith(".json"):
                with open(f"{game_dir}/{game_file}") as f:
                    game = json.load(f)
                
                game_name = game_file.replace(".json", "")
                
                # Walk the scene graph looking for high-ethical-cost choices
                for scene_id, scene in game.get("scenes", {}).items():
                    for i, choice in enumerate(scene.get("choices", [])):
                        morality = choice.get("morality", {})
                        
                        # Filter: at least one ethical violation > 0.5
                        violations = {k: v for k, v in morality.items() if v > 0.5}
                        if violations:
                            decision_nodes.append({
                                "id": f"mach-{game_name}-{scene_id}-c{i}",
                                "game": game_name,
                                "scene_id": scene_id,
                                "scene_text": scene.get("text", ""),
                                "choice_text": choice.get("text", ""),
                                "reward": choice.get("reward", 0),
                                "morality": morality,
                                "violations": violations,
                                "violation_categories": list(violations.keys()),
                                "num_violations": len(violations),
                            })
    
    # Sort by number of violations (multi-violation nodes are most useful)
    decision_nodes.sort(key=lambda x: (-x["num_violations"], -max(x["morality"].values(), default=0)))
    
    with open(f"{output_dir}/decision_nodes.json", "w") as f:
        json.dump(decision_nodes, f, indent=2)
    
    print(f"Extracted {len(decision_nodes)} high-ethical-cost decision nodes")
    print(f"Multi-violation nodes (2+): {sum(1 for n in decision_nodes if n['num_violations'] >= 2)}")
    return decision_nodes