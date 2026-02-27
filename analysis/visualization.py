# analysis/visualization.py
"""
Generate all paper figures from computed metrics.

Figures:
  1. CDIM Heatmap — asymmetric interference matrix
  2. SIS Distribution — by operator type
  3. CNI Scatter — predicted vs actual triple scores
  4. RSI vs SIS Scatter — separating real safety from over-refusal
  5. Per-model Radar — dimension scores per model
"""
import json, os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")


def generate_all_figures(results_dir: str = "analysis", output_dir: str = "figures"):
    """Generate all paper figures from precomputed metrics."""
    if not HAS_PLOTTING:
        print("Skipping figure generation (matplotlib/seaborn not available)")
        return

    os.makedirs(output_dir, exist_ok=True)

    _fig_cdim_heatmap(results_dir, output_dir)
    _fig_sis_distribution(results_dir, output_dir)
    _fig_cni_scatter(results_dir, output_dir)
    _fig_rsi_vs_sis(results_dir, output_dir)

    print(f"All figures saved to {output_dir}/")


def _fig_cdim_heatmap(results_dir: str, output_dir: str):
    """Fig 1: CDIM heatmap — the money figure."""
    path = os.path.join(results_dir, "cdim_matrix.json")
    if not os.path.exists(path):
        print("  Skipping CDIM heatmap (no data)")
        return

    with open(path) as f:
        cdim = json.load(f)

    dims = sorted(cdim.keys())
    matrix = np.array([[cdim.get(a, {}).get(b, 0) for b in dims] for a in dims])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=dims, yticklabels=dims,
                cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                square=True, ax=ax)
    ax.set_title("Cross-Dimension Interference Matrix (CDIM)")
    ax.set_xlabel("Affected Dimension (B)")
    ax.set_ylabel("Interfering Dimension (A)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_cdim_heatmap.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig1_cdim_heatmap.png"), dpi=300)
    plt.close()
    print("  Generated: fig1_cdim_heatmap")


def _fig_sis_distribution(results_dir: str, output_dir: str):
    """Fig 2: SIS distribution by operator type."""
    path = os.path.join(results_dir, "sis_results.json")
    if not os.path.exists(path):
        print("  Skipping SIS distribution (no data)")
        return

    with open(path) as f:
        sis = json.load(f)

    # Load tier2 scenarios to get operator labels
    t2_path = "scenarios/tier2/tier2_scenarios.json"
    pair_operators = {}
    if os.path.exists(t2_path):
        with open(t2_path) as f:
            t2 = json.load(f)
        for s in t2:
            pair_operators[s["pair_id"]] = s["operator"]

    # Organize SIS values by operator
    operator_sis = {}
    for pair_id_str, dims in sis.items():
        pair_id = int(pair_id_str) if pair_id_str.isdigit() else pair_id_str
        op = pair_operators.get(pair_id, "unknown")
        if op not in operator_sis:
            operator_sis[op] = []
        operator_sis[op].extend(dims.values())

    if not operator_sis:
        print("  Skipping SIS distribution (empty data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    operators = sorted(operator_sis.keys())
    data = [operator_sis[op] for op in operators]

    parts = ax.violinplot(data, positions=range(len(operators)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(operators)))
    ax.set_xticklabels(operators)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Composition Operator")
    ax.set_ylabel("Safety Interaction Score (SIS)")
    ax.set_title("SIS Distribution by Operator Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_sis_distribution.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig2_sis_distribution.png"), dpi=300)
    plt.close()
    print("  Generated: fig2_sis_distribution")


def _fig_cni_scatter(results_dir: str, output_dir: str):
    """Fig 3: CNI scatter — predicted vs actual triple scores."""
    path = os.path.join(results_dir, "cni_results.json")
    if not os.path.exists(path):
        print("  Skipping CNI scatter (no data)")
        return

    with open(path) as f:
        cni = json.load(f)

    if not cni:
        print("  Skipping CNI scatter (empty data)")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    configs = sorted(cni.keys())
    values = [cni[c] for c in configs]

    ax.bar(range(len(configs)), values, color=["#e74c3c" if v < 0 else "#2ecc71" for v in values])
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xlabel("Triple Config ID")
    ax.set_ylabel("Composition Nonlinearity Index (CNI)")
    ax.set_title("CNI: Nonlinear Compounding Effects")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_cni.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig3_cni.png"), dpi=300)
    plt.close()
    print("  Generated: fig3_cni")


def _fig_rsi_vs_sis(results_dir: str, output_dir: str):
    """Fig 4: RSI vs SIS scatter — separating genuine safety from over-refusal."""
    sis_path = os.path.join(results_dir, "sis_results.json")
    rsi_path = os.path.join(results_dir, "rsi_results.json")

    if not os.path.exists(sis_path) or not os.path.exists(rsi_path):
        print("  Skipping RSI vs SIS scatter (no data)")
        return

    with open(sis_path) as f:
        sis = json.load(f)
    with open(rsi_path) as f:
        rsi = json.load(f)

    # Match pairs
    pairs_x, pairs_y, labels = [], [], []
    for pair_id_str in sis:
        pair_id = int(pair_id_str) if pair_id_str.isdigit() else pair_id_str
        # Average SIS across dimensions for this pair
        dim_values = list(sis[pair_id_str].values())
        avg_sis = sum(dim_values) / len(dim_values) if dim_values else 0
        rsi_val = rsi.get(str(pair_id), rsi.get(pair_id, None))
        if rsi_val is not None:
            pairs_x.append(avg_sis)
            pairs_y.append(rsi_val)
            labels.append(str(pair_id))

    if not pairs_x:
        print("  Skipping RSI vs SIS scatter (no matching data)")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pairs_x, pairs_y, s=80, alpha=0.7, edgecolors="black", linewidth=0.5)

    for x, y, label in zip(pairs_x, pairs_y, labels):
        ax.annotate(f"P{label}", (x, y), fontsize=8, ha="center", va="bottom")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Quadrant labels
    ax.text(0.95, 0.95, "Safer + Over-refusing", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="orange", alpha=0.7)
    ax.text(0.95, 0.05, "Genuinely Safer", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="green", alpha=0.7)
    ax.text(0.05, 0.95, "Degraded + Over-refusing", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="red", alpha=0.7)
    ax.text(0.05, 0.05, "Degraded + Compliant", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="darkred", alpha=0.7)

    ax.set_xlabel("Safety Interaction Score (SIS)")
    ax.set_ylabel("Refusal Sensitivity Index (RSI)")
    ax.set_title("SIS vs RSI: Real Safety vs Over-Caution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_rsi_vs_sis.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig4_rsi_vs_sis.png"), dpi=300)
    plt.close()
    print("  Generated: fig4_rsi_vs_sis")
