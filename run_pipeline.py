# run_pipeline.py
"""
Full ATLAS v2 pipeline: download -> normalize -> compose -> validate -> evaluate.
"""
import argparse, os
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="ATLAS v2 pipeline")
    parser.add_argument("--step", choices=["download", "normalize", "tier1", "tier2",
                                           "tier3", "validate", "evaluate", "analyze", "all"])
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Cache atoms so normalize_all() runs at most once per invocation
    _atoms = None

    def get_atoms():
        nonlocal _atoms
        if _atoms is None:
            from processing.normalize import normalize_all
            _atoms = normalize_all()
        return _atoms

    if args.step in ("download", "all"):
        print("=== STEP 1: Download benchmarks ===")
        from download_all import main as download_main
        download_main()

    if args.step in ("normalize", "all"):
        print("=== STEP 2: Normalize to atoms ===")
        get_atoms()

    if args.step in ("tier1", "all"):
        print("=== STEP 3: Build Tier 1 ===")
        from composition.build_tier1 import build_tier1
        build_tier1(get_atoms(), seed=args.seed)

    if args.step in ("tier2", "all"):
        print("=== STEP 4: Build Tier 2 (GPT-4o composition) ===")
        from composition.build_tier2 import build_tier2
        build_tier2(get_atoms(), seed=args.seed)

    if args.step in ("tier3", "all"):
        print("=== STEP 5: Build Tier 3 ===")
        import json
        from composition.build_tier3 import build_tier3
        with open("scenarios/tier2/tier2_scenarios.json") as f:
            tier2 = json.load(f)
        build_tier3(tier2, get_atoms(), seed=args.seed)

    if args.step in ("validate", "all"):
        print("=== STEP 6: Validate ===")
        import json
        from composition.validate import validate_dataset
        with open("scenarios/tier1/tier1_scenarios.json") as f:
            t1 = json.load(f)
        with open("scenarios/tier2/tier2_scenarios.json") as f:
            t2 = json.load(f)
        with open("scenarios/tier3/tier3_scenarios.json") as f:
            t3 = json.load(f)
        validate_dataset(t1, t2, t3)

    if args.step in ("evaluate", "all"):
        print(f"=== STEP 7: Evaluate with {args.model} ===")
        import subprocess
        cmd = ["inspect", "eval", "pipeline/to_inspect.py", "--model", args.model]
        if args.tier:
            cmd.extend(["-T", f"tier={args.tier}"])
        subprocess.run(cmd, check=True)

    if args.step in ("analyze", "all"):
        print("=== STEP 8: Compute metrics + figures ===")
        from analysis.sis_computation import compute_all_sis
        from analysis.cdim_matrix import compute_cdim
        from analysis.cni_computation import compute_all_cni
        from analysis.rsi_computation import compute_all_rsi
        from analysis.visualization import generate_all_figures

        log_dir = os.path.join(os.getcwd(), "logs")
        compute_all_sis(log_dir)
        compute_cdim(log_dir)
        compute_all_cni(log_dir)
        compute_all_rsi(log_dir)
        generate_all_figures()


if __name__ == "__main__":
    main()
