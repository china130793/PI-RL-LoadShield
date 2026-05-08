from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

from loadshield_data import (
    ABLATION_COLORS,
    ABLATION_ORDER,
    CONTROLLER_COLORS,
    CONTROLLER_LABELS,
    CONTROLLER_ORDER,
    SEVERITY_INDEX,
    SEVERITY_ORDER,
    derive_ablation_table,
    mean_by_controller,
    require_columns,
    robustness_grid,
    warn,
)


def _representative_episode(ts: pd.DataFrame, scenario_id: str = "S4") -> pd.DataFrame:
    subset = ts[ts["scenario_id"] == scenario_id].copy()
    if subset.empty:
        warn(f"Scenario {scenario_id} missing; using the first available scenario instead.")
        scenario_id = str(ts["scenario_id"].iloc[0])
        subset = ts[ts["scenario_id"] == scenario_id].copy()
    return subset


def plot_time_response(ts: pd.DataFrame, out_dir: Path) -> None:
    required = [
        "scenario_id", "controller_type", "time_s", "rotor_arrival_wind_speed_mps",
        "M_tb_kNm", "M_br_1_kNm", "M_br_2_kNm", "M_br_3_kNm", "T_m_kN",
        "T_m_safe_kN", "delta_beta_1_deg", "delta_beta_2_deg", "delta_beta_3_deg",
        "delta_yaw_deg", "yaw_misalignment_deg", "gust_start_s"
    ]
    if require_columns(ts, required, "loadshield_timeseries_dataset.csv"):
        warn("Skipping Figure 4 because required time-series columns are missing.")
        return

    data = _representative_episode(ts, "S4")
    pi = data[data["controller_type"] == "PI-RL LoadShield"].sort_values("time_s")
    collective = data[data["controller_type"] == "Collective Pitch"].sort_values("time_s")
    mfdrl = data[data["controller_type"] == "Model-free DRL"].sort_values("time_s")
    if pi.empty:
        warn("Skipping Figure 4 because PI-RL LoadShield rows are missing.")
        return

    preview_cols = [
        "lidar_preview_wind_speed_30s_mps",
        "preview_wind_speed_30s_mps",
        "preview_wind_speed",
        "lidar_preview_wind_speed_20s_mps",
        "lidar_preview_wind_speed_10s_mps",
    ]
    preview_col = next((c for c in preview_cols if c in pi.columns), None)
    if not preview_col:
        warn("Skipping Figure 4 because no preview wind-speed column was found.")
        return

    fig = plt.figure(figsize=(8.2, 10.0))
    gs = gridspec.GridSpec(6, 1, hspace=0.12)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]
    t = pi["time_s"].to_numpy()
    gust_start = float(pi["gust_start_s"].iloc[0])
    event_markers = [
        (gust_start - 30.0, "Preview detection"),
        (gust_start, "Rotor impact"),
        (float(pi.loc[pi["M_tb_kNm"].abs().idxmax(), "time_s"]), "Peak response"),
    ]

    axes[0].plot(t, pi[preview_col], "--", lw=1.4, label="LiDAR preview")
    axes[0].plot(t, pi["rotor_arrival_wind_speed_mps"], "-", lw=1.4, label="Rotor-arrival wind")
    axes[0].set_ylabel("Wind\n(m/s)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8, frameon=False)

    norm_source = collective["M_tb_kNm"].abs().max() if not collective.empty else pi["M_tb_kNm"].abs().max()
    for ctrl_df, label in [(collective, "Collective pitch"), (mfdrl, "Model-free DRL"), (pi, "PI-RL LoadShield")]:
        if not ctrl_df.empty:
            axes[1].plot(ctrl_df["time_s"], ctrl_df["M_tb_kNm"] / norm_source, lw=1.3, label=label)
    axes[1].set_ylabel("$M_{tb}$\n(norm.)")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)

    for col, label in [("M_br_1_kNm", "$M_{br,1}$"), ("M_br_2_kNm", "$M_{br,2}$"), ("M_br_3_kNm", "$M_{br,3}$")]:
        axes[2].plot(t, pi[col], lw=1.2, label=label)
    br_min = pi[["M_br_1_kNm", "M_br_2_kNm", "M_br_3_kNm"]].min(axis=1)
    br_max = pi[["M_br_1_kNm", "M_br_2_kNm", "M_br_3_kNm"]].max(axis=1)
    axes[2].fill_between(t, br_min, br_max, alpha=0.12)
    axes[2].set_ylabel("$M_{br}$\n(kNm)")
    axes[2].legend(loc="upper right", ncol=3, fontsize=8, frameon=False)

    if not collective.empty:
        axes[3].plot(collective["time_s"], collective["T_m_kN"], lw=1.2, label="Collective pitch")
    axes[3].plot(t, pi["T_m_kN"], lw=1.3, label="PI-RL LoadShield")
    axes[3].axhline(pi["T_m_safe_kN"].iloc[0], ls="--", lw=1.1, color="black", label="$T_m^{safe}$")
    axes[3].set_ylabel("$T_m$\n(kN)")
    axes[3].legend(loc="upper right", ncol=3, fontsize=8, frameon=False)

    for col, label in [("delta_beta_1_deg", r"$\Delta\beta_1$"), ("delta_beta_2_deg", r"$\Delta\beta_2$"), ("delta_beta_3_deg", r"$\Delta\beta_3$")]:
        axes[4].plot(t, pi[col], lw=1.2, label=label)
    axes[4].set_ylabel(r"$\Delta\beta$" + "\n(deg)")
    axes[4].legend(loc="upper right", ncol=3, fontsize=8, frameon=False)

    axes[5].plot(t, pi["delta_yaw_deg"], lw=1.3, label=r"$\Delta\psi_y$")
    axes[5].plot(t, pi["yaw_misalignment_deg"], lw=1.2, ls="--", label="Yaw misalignment")
    axes[5].set_ylabel("Yaw\n(deg)")
    axes[5].set_xlabel("Time (s)")
    axes[5].legend(loc="upper right", ncol=2, fontsize=8, frameon=False)

    for ax in axes:
        ax.grid(True, lw=0.35, alpha=0.4)
        ax.set_xlim(t.min(), t.max())
        for x, _ in event_markers:
            if t.min() <= x <= t.max():
                ax.axvline(x, color="0.25", lw=0.8, ls=":")
        ax.tick_params(labelsize=8)
    ymax = axes[0].get_ylim()[1]
    for x, label in event_markers:
        if t.min() <= x <= t.max():
            axes[0].text(x, ymax, label, rotation=90, va="top", ha="right", fontsize=7)

    axes[0].set_title("Figure 4. Time-domain response under a combined gust-wave event", fontsize=11)
    fig.savefig(out_dir / "figures" / "figure4_time_domain_response.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figures" / "figure4_time_domain_response.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_feasibility_dashboard(summary: pd.DataFrame, out_dir: Path) -> None:
    required = [
        "controller_type", "Mtb_peak_reduction_pct", "Mbr_rms_reduction_pct",
        "Tm_peak_reduction_pct", "DEL_reduction_pct", "eP_RMS",
        "N_viol_applied", "A_beta", "A_psi", "S_a", "an_rms_reduction_pct", "L_inf_ms"
    ]
    if require_columns(summary, required, "episode_summary_metrics.csv"):
        warn("Skipping Figure 5 because required summary columns are missing.")
        return

    avg = mean_by_controller(summary).set_index("controller_type").reindex(CONTROLLER_ORDER).reset_index()
    fig = plt.figure(figsize=(11.2, 8.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    components = ["Tower-base", "Blade-root", "Mooring"]
    metrics = ["Mtb_peak_reduction_pct", "Mbr_rms_reduction_pct", "Tm_peak_reduction_pct"]
    x = np.arange(len(components))
    width = 0.14
    for i, ctrl in enumerate(CONTROLLER_ORDER):
        vals = []
        for m in metrics:
            val = float(avg.loc[avg["controller_type"] == ctrl, m].iloc[0])
            vals.append(0.0 if ctrl == "Collective Pitch" else val)
        ax1.bar(x + (i - 2) * width, vals, width, label=CONTROLLER_LABELS[ctrl], color=CONTROLLER_COLORS[ctrl], alpha=0.88, edgecolor="black", linewidth=0.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontsize=8)
    ax1.set_ylabel("Reduction (%)")
    ax1.set_title("(a) Structural-load reductions", fontsize=10)
    ax1.grid(axis="y", lw=0.35, alpha=0.4)

    ax2 = fig.add_subplot(gs[0, 1])
    for _, row in avg.iterrows():
        ctrl = row["controller_type"]
        y = 0.0 if ctrl == "Collective Pitch" else float(row["DEL_reduction_pct"])
        size = 45 + 24 * float(row["N_viol_applied"])
        ax2.scatter(row["eP_RMS"], y, s=size, color=CONTROLLER_COLORS[ctrl], edgecolor="black", linewidth=0.35, alpha=0.88)
        ax2.text(row["eP_RMS"] + 0.00035, y + 0.4, CONTROLLER_LABELS[ctrl], fontsize=7)
    ax2.set_xlabel("$e_P^{RMS}$")
    ax2.set_ylabel("DEL reduction (%)")
    ax2.set_title("(b) Power stability vs. fatigue reduction", fontsize=10)
    ax2.grid(True, lw=0.35, alpha=0.4)

    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    radar_metrics = ["A_beta", "A_psi", "S_a"]
    radar_labels = [r"$A_\beta$", r"$A_\psi$", r"$S_a$"]
    theta = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    max_vals = avg[radar_metrics].max().replace(0, 1.0)
    for _, row in avg.iterrows():
        ctrl = row["controller_type"]
        vals = (row[radar_metrics] / max_vals).astype(float).to_numpy()
        vals = np.concatenate([vals, vals[:1]])
        ax3.plot(theta, vals, lw=1.2, color=CONTROLLER_COLORS[ctrl], label=CONTROLLER_LABELS[ctrl])
        ax3.fill(theta, vals, color=CONTROLLER_COLORS[ctrl], alpha=0.06)
    ax3.set_xticks(theta[:-1])
    ax3.set_xticklabels(radar_labels, fontsize=8)
    ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax3.set_title("(c) Normalized actuator effort", fontsize=10, pad=12)

    ax4 = fig.add_subplot(gs[1, 1])
    labels = [CONTROLLER_LABELS[c] for c in CONTROLLER_ORDER]
    xpos = np.arange(len(labels))
    an_vals = []
    latency_vals = []
    for ctrl in CONTROLLER_ORDER:
        row = avg[avg["controller_type"] == ctrl].iloc[0]
        an_vals.append(0.0 if ctrl == "Collective Pitch" else row["an_rms_reduction_pct"])
        latency_vals.append(row["L_inf_ms"])
    ax4.bar(xpos, an_vals, color=[CONTROLLER_COLORS[c] for c in CONTROLLER_ORDER], edgecolor="black", linewidth=0.25)
    ax4.set_ylabel("$a_n^{RMS}$ reduction (%)")
    ax4.set_xticks(xpos)
    ax4.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax4.grid(axis="y", lw=0.35, alpha=0.4)
    ax4b = ax4.twinx()
    ax4b.plot(xpos, latency_vals, color="black", marker="o", lw=1.25)
    ax4b.set_ylabel("$L_{inf}$ (ms)")
    ax4.set_title("(d) Nacelle acceleration reduction and latency", fontsize=10)

    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Figure 5. Fatigue-power-actuator feasibility dashboard", fontsize=12, y=0.985)
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(out_dir / "figures" / "figure5_feasibility_dashboard.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figures" / "figure5_feasibility_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(summary: pd.DataFrame, ablation: Optional[pd.DataFrame], out_dir: Path) -> None:
    abl = derive_ablation_table(summary, ablation)
    if abl is None or abl.empty:
        warn("Skipping Figure 6 because complete ablation data is not available.")
        return

    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    metrics = ["M_tb_peak", "Gamma_br", "T_m_peak", "DEL", "N_viol"]
    labels = ["$M_{tb}^{peak}$\nreduction (%)", "$\\Gamma_{br}$\nreduction (%)", "$T_m^{peak}$\nreduction (%)", "DEL\nreduction (%)", "$N_{viol}$\n(count)"]
    x = np.arange(len(metrics))
    max_reduction = max(60.0, float(abl[metrics[:4]].max().max()) * 1.15)
    max_viol = max(1.0, float(abl["N_viol"].max()))

    full = abl[abl["Variant"] == "Full PI-RL LoadShield"]
    if not full.empty:
        full = full.iloc[0]
        for i, metric in enumerate(metrics[:4]):
            ax.axhspan(float(full[metric]), max_reduction, xmin=(i - 0.45) / len(metrics), xmax=(i + 0.45) / len(metrics), color="#dff1dc", alpha=0.75)
        ax.axhspan(0, max_reduction * 0.2, xmin=(4 - 0.45) / len(metrics), xmax=(4 + 0.45) / len(metrics), color="#dff1dc", alpha=0.75)

    for _, row in abl.iterrows():
        variant = row["Variant"]
        raw = [row[m] for m in metrics]
        y = raw.copy()
        y[-1] = max_reduction - (float(row["N_viol"]) / max_viol) * max_reduction
        ax.plot(x, y, marker="o", lw=1.7 if variant == "Full PI-RL LoadShield" else 1.15, color=ABLATION_COLORS.get(variant, "0.35"), label=variant, alpha=0.95)
        for xi, yi, val in zip(x, y, raw):
            text = f"{int(val)}" if xi == 4 else f"{float(val):.1f}"
            ax.text(xi + 0.025, yi + 0.8, text, fontsize=7, color=ABLATION_COLORS.get(variant, "0.35"))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Metric scale / desirability-aligned safety scale")
    ax.grid(True, axis="y", lw=0.35, alpha=0.45)
    ax.set_title("Figure 6. Unified ablation-performance plot of PI-RL LoadShield", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.savefig(out_dir / "figures" / "figure6_unified_ablation_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figures" / "figure6_unified_ablation_plot.pdf", bbox_inches="tight")
    plt.close(fig)


def _matrix_from_grid(grid: pd.DataFrame, col: str, severities: List[str], delays: List[int]) -> np.ndarray:
    mat = np.full((len(severities), len(delays)), np.nan)
    for i, sev in enumerate(severities):
        for j, delay in enumerate(delays):
            row = grid[(grid["storm_severity"] == sev) & (grid["communication_delay_ms"].round().astype(int) == int(delay))]
            if not row.empty:
                mat[i, j] = float(row[col].mean())
    return mat


def _heatmap(ax, matrix, xlabels, ylabels, cmap, vmin, vmax, fmt, title):
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Communication delay (ms)", fontsize=9)
    ax.set_ylabel("Storm severity", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, "NA" if np.isnan(value) else format(value, fmt), ha="center", va="center", fontsize=8)
    return image


def plot_robustness(summary: pd.DataFrame, robustness: Optional[pd.DataFrame], out_dir: Path) -> None:
    grid = robustness_grid(summary, robustness)
    if grid is None or grid.empty:
        warn("Skipping Figure 7 because robustness data is not available.")
        return

    delays = sorted([int(round(x)) for x in grid["communication_delay_ms"].dropna().unique()])
    severities = [s for s in SEVERITY_ORDER if s in set(grid["storm_severity"].unique())]
    if not delays or not severities:
        warn("Skipping Figure 7 because delay or severity values are unavailable.")
        return

    mtb = _matrix_from_grid(grid, "Mtb_peak_normalized", severities, delays)
    nviol = _matrix_from_grid(grid, "N_viol_pre", severities, delays)
    margin = _matrix_from_grid(grid, "normalized_safety_margin", severities, delays)

    fig = plt.figure(figsize=(11.5, 8.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.25, height_ratios=[1.0, 1.15])
    ylabels = [f"{s}\n({SEVERITY_INDEX.get(s, i + 1)})" for i, s in enumerate(severities)]

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = _heatmap(ax1, mtb, delays, ylabels, "coolwarm", float(np.nanmin(mtb)), float(np.nanmax(mtb)), ".2f", "(a) Normalized peak tower-base moment")
    fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.03)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = _heatmap(ax2, nviol, delays, ylabels, "RdYlBu_r", 0, max(1.0, float(np.nanmax(nviol))), ".0f", "(b) Pre-projection safety violations")
    fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)

    ax3 = fig.add_subplot(gs[1, :])
    vmin = min(-0.01, float(np.nanmin(margin)))
    vmax = max(0.01, float(np.nanmax(margin)))
    im3 = ax3.imshow(margin, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), aspect="auto", origin="lower")
    ax3.set_xticks(np.arange(len(delays)))
    ax3.set_xticklabels(delays, fontsize=8)
    ax3.set_yticks(np.arange(len(severities)))
    ax3.set_yticklabels([SEVERITY_INDEX.get(s, i + 1) for i, s in enumerate(severities)], fontsize=8)
    ax3.set_xlabel("Communication delay (ms)", fontsize=9)
    ax3.set_ylabel("Storm severity index", fontsize=9)
    ax3.set_title("(c) Normalized safety margin", fontsize=10)
    ax3.set_xticks(np.arange(-0.5, len(delays), 1), minor=True)
    ax3.set_yticks(np.arange(-0.5, len(severities), 1), minor=True)
    ax3.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax3.tick_params(which="minor", bottom=False, left=False)
    for i in range(margin.shape[0]):
        for j in range(margin.shape[1]):
            value = margin[i, j]
            ax3.text(j, i, "NA" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=8)
    if np.nanmin(margin) <= 0.0 <= np.nanmax(margin):
        ax3.contour(margin, levels=[0.0], colors="black", linewidths=1.8, origin="lower")
    fig.colorbar(im3, ax=ax3, fraction=0.025, pad=0.02)
    fig.suptitle("Figure 7. Robustness analysis under communication delay and storm severity", fontsize=12)
    fig.savefig(out_dir / "figures" / "figure7_robustness_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "figures" / "figure7_robustness_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(ts: pd.DataFrame, summary: pd.DataFrame, ablation: Optional[pd.DataFrame], robustness: Optional[pd.DataFrame], out_dir: Path) -> None:
    plot_time_response(ts, out_dir)
    plot_feasibility_dashboard(summary, out_dir)
    plot_ablation(summary, ablation, out_dir)
    plot_robustness(summary, robustness, out_dir)
