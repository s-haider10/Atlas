# analysis/visualization.py
"""
Generate all paper figures from computed metrics.

Figures:
  1. CDIM Heatmap — asymmetric interference matrix with null cells grayed out
  2. SIS Distribution — by operator type with CI error bars
  3. CNI Bar Chart — with CI error bars and significance markers
  4. RSI vs SIS Scatter — separating real safety from over-refusal
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


def generate_all_figures(results_dir: str = "analysis", output_dir: str = "figures",
                         run_label: str = None):
    """Generate all paper figures from precomputed metrics.

    Args:
        results_dir: Directory containing computed metric JSONs.
        output_dir: Base output directory for figures.
        run_label: Optional label (e.g. model name) to create a subfolder.
            If provided, figures are saved to output_dir/run_label/.
            This prevents overwriting plots from previous analysis runs.
    """
    if not HAS_PLOTTING:
        print("Skipping figure generation (matplotlib/seaborn not available)")
        return

    if run_label:
        # Sanitize label for use as directory name
        safe_label = run_label.replace("/", "_").replace("\\", "_")
        output_dir = os.path.join(output_dir, safe_label)

    os.makedirs(output_dir, exist_ok=True)

    _fig_cdim_heatmap(results_dir, output_dir)
    _fig_sis_distribution(results_dir, output_dir)
    _fig_cni_bar(results_dir, output_dir)
    _fig_rsi_vs_sis(results_dir, output_dir)

    print(f"All figures saved to {output_dir}/")


def _fig_cdim_heatmap(results_dir: str, output_dir: str):
    """Fig 1: CDIM heatmap with null cells grayed out and sample counts annotated."""
    path = os.path.join(results_dir, "cdim_matrix.json")
    if not os.path.exists(path):
        print("  Skipping CDIM heatmap (no data)")
        return

    with open(path) as f:
        data = json.load(f)

    # Handle new format (nested with matrix/counts/pvalues) or old format (flat dict)
    if "matrix" in data:
        cdim = data["matrix"]
        counts = data.get("counts", {})
    else:
        cdim = data
        counts = {}

    dims = sorted(cdim.keys())
    n = len(dims)

    # Build numeric matrix and mask for null/not_measurable cells
    matrix = np.full((n, n), np.nan)
    annot = np.full((n, n), "", dtype=object)
    mask = np.full((n, n), False)

    for i, da in enumerate(dims):
        for j, db in enumerate(dims):
            val = cdim.get(da, {}).get(db)
            cnt = counts.get(da, {}).get(db, 0)
            if val is None or val == "not_measurable":
                mask[i][j] = True
                annot[i][j] = "n/a" if da == db else "—"
            else:
                matrix[i][j] = float(val)
                annot[i][j] = f"{val:.2f}\nn={cnt}"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, xticklabels=dims, yticklabels=dims,
                cmap="RdBu_r", center=0, annot=annot, fmt="",
                square=True, ax=ax, mask=mask,
                cbar_kws={"label": "CDIM Score"})

    # Gray out masked cells
    for i in range(n):
        for j in range(n):
            if mask[i][j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                            color="#e0e0e0", zorder=0))

    ax.set_title("Cross-Dimension Interference Matrix (CDIM)")
    ax.set_xlabel("Affected Dimension (B)")
    ax.set_ylabel("Interfering Dimension (A)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_cdim_heatmap.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig1_cdim_heatmap.png"), dpi=300)
    plt.close()
    print("  Generated: fig1_cdim_heatmap")


def _fig_sis_distribution(results_dir: str, output_dir: str):
    """Fig 2: SIS distribution by operator type with CI error bars."""
    path = os.path.join(results_dir, "sis_results.json")
    if not os.path.exists(path):
        print("  Skipping SIS distribution (no data)")
        return

    with open(path) as f:
        sis = json.load(f)

    # Load tier2 scenarios to get operator labels
    t2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "scenarios/tier2/tier2_scenarios.json")
    pair_operators = {}
    if os.path.exists(t2_path):
        with open(t2_path) as f:
            t2 = json.load(f)
        for s in t2:
            pair_operators[s["pair_id"]] = s["operator"]

    # Organize SIS values by operator (extract numeric SIS, skip nulls)
    operator_sis = {}
    for pair_id_str, dims in sis.items():
        pair_id = int(pair_id_str) if pair_id_str.isdigit() else pair_id_str
        op = pair_operators.get(pair_id, "unknown")
        if op not in operator_sis:
            operator_sis[op] = []
        for dim_data in dims.values():
            val = dim_data.get("sis") if isinstance(dim_data, dict) else dim_data
            if val is not None:
                operator_sis[op].append(val)

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

    # Add mean ± CI labels
    for i, (op, vals) in enumerate(zip(operators, data)):
        if vals:
            mean = sum(vals) / len(vals)
            ax.annotate(f"μ={mean:.2f}\nn={len(vals)}", (i, mean),
                        fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("Composition Operator")
    ax.set_ylabel("Safety Interaction Score (SIS)")
    ax.set_title("SIS Distribution by Operator Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_sis_distribution.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig2_sis_distribution.png"), dpi=300)
    plt.close()
    print("  Generated: fig2_sis_distribution")


def _fig_cni_bar(results_dir: str, output_dir: str):
    """Fig 3: CNI bar chart with CI error bars and significance markers."""
    path = os.path.join(results_dir, "cni_results.json")
    if not os.path.exists(path):
        print("  Skipping CNI bar chart (no data)")
        return

    with open(path) as f:
        cni = json.load(f)

    if not cni:
        print("  Skipping CNI bar chart (empty data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    configs = sorted(cni.keys(), key=lambda x: int(x) if str(x).isdigit() else x)

    x_pos = range(len(configs))
    values = []
    errors = []
    colors = []
    sig_markers = []

    for c in configs:
        entry = cni[c]
        if isinstance(entry, dict):
            val = entry.get("cni")
            ci = entry.get("ci_95")
            sig = entry.get("significant", False)
        else:
            val = entry
            ci = None
            sig = False

        if val is None:
            values.append(0)
            errors.append(0)
            colors.append("#cccccc")
            sig_markers.append("")
        else:
            values.append(val)
            if ci:
                errors.append(val - ci[0])  # distance from mean to lower bound
            else:
                errors.append(0)
            colors.append("#e74c3c" if val < 0 else "#2ecc71")
            sig_markers.append("*" if sig else "")

    bars = ax.bar(x_pos, values, yerr=errors, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)

    # Add significance markers
    for i, (bar, marker) in enumerate(zip(bars, sig_markers)):
        if marker:
            height = bar.get_height()
            y_pos = height + errors[i] + 0.02 if height >= 0 else height - errors[i] - 0.05
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, marker,
                    ha="center", va="bottom", fontsize=14, fontweight="bold")

    # Add n labels
    for i, c in enumerate(configs):
        entry = cni[c]
        n = entry.get("n", "?") if isinstance(entry, dict) else "?"
        ax.text(i, -0.02, f"n={n}", ha="center", va="top", fontsize=7,
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xlabel("Triple Config ID")
    ax.set_ylabel("Composition Nonlinearity Index (CNI)")
    ax.set_title("CNI: Nonlinear Compounding Effects (* = p<0.05)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_cni.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig3_cni.png"), dpi=300)
    plt.close()
    print("  Generated: fig3_cni")


def _fig_rsi_vs_sis(results_dir: str, output_dir: str):
    """Fig 4: RSI vs SIS scatter — separating genuine safety from over-refusal.

    Plots one point PER DIMENSION (not per pair) to avoid hiding opposing
    effects when averaging. Each point is a (pair, dimension) combination.
    Marker shape distinguishes dimension A from dimension B.
    """
    sis_path = os.path.join(results_dir, "sis_results.json")
    rsi_path = os.path.join(results_dir, "rsi_results.json")

    if not os.path.exists(sis_path) or not os.path.exists(rsi_path):
        print("  Skipping RSI vs SIS scatter (no data)")
        return

    with open(sis_path) as f:
        sis = json.load(f)
    with open(rsi_path) as f:
        rsi = json.load(f)

    # Plot per-dimension points (not averaged)
    points_x, points_y, labels, sig_flags, dim_indices = [], [], [], [], []
    for pair_id_str, dims_data in sis.items():
        pair_id = int(pair_id_str) if pair_id_str.isdigit() else pair_id_str

        rsi_entry = rsi.get(str(pair_id), rsi.get(pair_id))
        if rsi_entry is None:
            continue
        rsi_val = rsi_entry.get("rsi") if isinstance(rsi_entry, dict) else rsi_entry
        if rsi_val is None:
            continue

        sig = rsi_entry.get("significant", False) if isinstance(rsi_entry, dict) else False

        for dim_idx, (dim_name, dim_data) in enumerate(dims_data.items()):
            if isinstance(dim_data, dict):
                sis_val = dim_data.get("sis")
            else:
                sis_val = dim_data
            if sis_val is None:
                continue

            points_x.append(sis_val)
            points_y.append(rsi_val)
            labels.append(f"P{pair_id}:{dim_name[:8]}")
            sig_flags.append(sig)
            dim_indices.append(dim_idx)

    if not points_x:
        print("  Skipping RSI vs SIS scatter (no matching data)")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # Different markers for dimension A vs B (vs C for Tier 3)
    markers = ["o", "s", "D", "^"]
    marker_labels = ["Dim A", "Dim B", "Dim C", "Dim D+"]

    for didx in sorted(set(dim_indices)):
        mask = [i for i, d in enumerate(dim_indices) if d == didx]
        if not mask:
            continue
        xs = [points_x[i] for i in mask]
        ys = [points_y[i] for i in mask]
        cs = ["#e74c3c" if sig_flags[i] else "#3498db" for i in mask]
        m = markers[min(didx, len(markers) - 1)]
        ax.scatter(xs, ys, s=70, alpha=0.7, c=cs, marker=m,
                   edgecolors="black", linewidth=0.5,
                   label=marker_labels[min(didx, len(marker_labels) - 1)])

    # Annotate points (offset to reduce overlap)
    for x, y, label in zip(points_x, points_y, labels):
        ax.annotate(label, (x, y), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

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

    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_xlabel("Safety Interaction Score (SIS) — per dimension")
    ax.set_ylabel("Refusal Sensitivity Index (RSI)")
    ax.set_title("SIS vs RSI: Per-Dimension View\n(red = significant RSI change, shape = dimension index)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_rsi_vs_sis.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig4_rsi_vs_sis.png"), dpi=300)
    plt.close()
    print("  Generated: fig4_rsi_vs_sis")
