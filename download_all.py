# download_all.py
"""
Run this once to download all 9 source benchmarks.
Estimated time: 5-10 minutes (mostly git clones).
Estimated disk: ~2-3 GB (MACHIAVELLI game data is largest).
"""
from benchmarks.download_agentharm import download_agentharm
from benchmarks.download_agentdojo import download_agentdojo
from benchmarks.download_tau_bench import download_tau_bench
from benchmarks.download_injecagent import download_injecagent
from benchmarks.download_machiavelli import download_machiavelli
from benchmarks.download_moral_stories import download_moral_stories
from benchmarks.download_ethics import download_ethics
from benchmarks.download_mic import download_mic
from benchmarks.download_darkbench import download_darkbench

if __name__ == "__main__":
    print("=" * 60)
    print("ATLAS v2 — Downloading all source benchmarks")
    print("=" * 60)
    
    benchmarks = {
        "agentharm": download_agentharm,
        "agentdojo": download_agentdojo,
        "tau_bench": download_tau_bench,
        "injecagent": download_injecagent,
        "machiavelli": download_machiavelli,
        "moral_stories": download_moral_stories,
        "ethics": download_ethics,
        "mic": download_mic,
        "darkbench": download_darkbench,
    }
    
    results = {}
    for name, download_fn in benchmarks.items():
        print(f"\n--- Downloading {name} ---")
        try:
            results[name] = download_fn()
            print(f"✓ {name} complete")
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            results[name] = None
    
    print("\n" + "=" * 60)
    print("Download Summary:")
    for name, result in results.items():
        status = "✓" if result is not None else "✗"
        count = len(result) if isinstance(result, list) else "N/A"
        print(f"  {status} {name}: {count} records")