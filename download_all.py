# download_all.py
"""
Run this once to download all 4 source benchmarks.
Estimated time: 3-5 minutes.
Estimated disk: ~500 MB.

Four benchmarks, each with a distinct composition role:
  - AgentHarm: harmful task refusal baseline (tool-use spine)
  - InjecAgent: injection resistance via cross-product assembly (attack layer)
  - Moral Stories: norm-grounded moral reasoning (value constraint layer)
  - DarkBench: dark pattern / manipulative behavior (pressure layer)
"""
from benchmarks.download_agentharm import download_agentharm
from benchmarks.download_injecagent import download_injecagent
from benchmarks.download_moral_stories import download_moral_stories
from benchmarks.download_darkbench import download_darkbench

def main():
    print("=" * 60)
    print("ATLAS v2 — Downloading all source benchmarks")
    print("=" * 60)

    benchmarks = {
        "agentharm": download_agentharm,
        "injecagent": download_injecagent,
        "moral_stories": download_moral_stories,
        "darkbench": download_darkbench,
    }

    results = {}
    for name, download_fn in benchmarks.items():
        print(f"\n--- Downloading {name} ---")
        try:
            results[name] = download_fn()
            print(f"  {name} complete")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None

    print("\n" + "=" * 60)
    print("Download Summary:")
    for name, result in results.items():
        status = "OK" if result is not None else "FAILED"
        if isinstance(result, list):
            count = len(result)
        elif isinstance(result, tuple):
            count = sum(len(r) for r in result if isinstance(r, (list, dict)))
        else:
            count = "N/A"
        print(f"  [{status}] {name}: {count} records")

if __name__ == "__main__":
    main()
