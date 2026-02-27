# run_pipeline.py
"""
Full ATLAS v2 pipeline: download -> normalize -> compose -> validate -> evaluate -> analyze.

Usage:
    # Full pipeline with default model
    uv run python run_pipeline.py --step all

    # Individual steps
    uv run python run_pipeline.py --step download
    uv run python run_pipeline.py --step tier2 --seed 42
    uv run python run_pipeline.py --step evaluate --model openai/gpt-4o-mini
    uv run python run_pipeline.py --step evaluate --model anthropic/claude-sonnet-4-20250514
    uv run python run_pipeline.py --step analyze --model openai/gpt-4o-mini

    # Evaluate a specific tier only
    uv run python run_pipeline.py --step evaluate --model openai/gpt-4o-mini --tier 2
"""
import argparse, os
from dotenv import load_dotenv
load_dotenv()


def _model_label(model: str) -> str:
    """Sanitize model name for use as a directory name."""
    return model.replace("/", "_").replace("\\", "_")


def _find_latest_eval_log(log_dir: str, model: str = None) -> str:
    """Find the most recent Inspect eval log directory.

    Inspect AI saves logs to ./logs/ by default. Each eval run creates
    a timestamped .eval file. We find the latest one, optionally filtered
    by model name.
    """
    if not os.path.exists(log_dir):
        return log_dir

    # Look for .eval files or subdirectories
    entries = []
    for f in os.listdir(log_dir):
        full = os.path.join(log_dir, f)
        if model and _model_label(model) not in f.lower() and model.split("/")[-1] not in f.lower():
            continue
        entries.append((os.path.getmtime(full), full))

    if entries:
        entries.sort(reverse=True)
        return entries[0][1]
    return log_dir


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS v2 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline:     uv run python run_pipeline.py --step all --model openai/gpt-4o-mini
  Download only:     uv run python run_pipeline.py --step download
  Compose Tier 2:    uv run python run_pipeline.py --step tier2
  Evaluate:          uv run python run_pipeline.py --step evaluate --model anthropic/claude-sonnet-4-20250514
  Analyze:           uv run python run_pipeline.py --step analyze --model openai/gpt-4o-mini
  Evaluate + analyze:uv run python run_pipeline.py --step evaluate --model openai/gpt-4o-mini
                     uv run python run_pipeline.py --step analyze --model openai/gpt-4o-mini

Supported models (examples):
  openai/gpt-4o-mini          openai/gpt-4o           openai/gpt-4.1
  anthropic/claude-sonnet-4-20250514   anthropic/claude-haiku-4-5-20251001
  together/meta-llama/Llama-3.1-70B-Instruct-Turbo
  together/Qwen/Qwen2.5-72B-Instruct-Turbo
  ollama/llama3.1              ollama/qwen2.5
""",
    )
    parser.add_argument("--step", required=True,
                        choices=["download", "normalize", "tier1", "tier2",
                                 "tier3", "validate", "evaluate", "analyze", "all"],
                        help="Pipeline step to run")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="Model to evaluate (Inspect AI format: provider/model)")
    parser.add_argument("--tier", type=int, default=None,
                        help="Evaluate a specific tier only (1, 2, or 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible scenario sampling")
    args = parser.parse_args()

    model_label = _model_label(args.model)

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
        print(f"=== STEP 8: Compute metrics + figures ({args.model}) ===")
        from analysis.sis_computation import compute_all_sis
        from analysis.cdim_matrix import compute_cdim
        from analysis.cni_computation import compute_all_cni
        from analysis.rsi_computation import compute_all_rsi
        from analysis.visualization import generate_all_figures

        # Per-model analysis output: analysis/<model_label>/
        analysis_dir = os.path.join("analysis", model_label)
        os.makedirs(analysis_dir, exist_ok=True)

        log_dir = os.path.join(os.getcwd(), "logs")
        compute_all_sis(log_dir)
        compute_cdim(log_dir)
        compute_all_cni(log_dir)
        compute_all_rsi(log_dir)

        # Figures go to figures/<model_label>/ so previous runs aren't overwritten
        generate_all_figures(run_label=model_label)

        print(f"\nResults: analysis/{model_label}/")
        print(f"Figures: figures/{model_label}/")


if __name__ == "__main__":
    main()
