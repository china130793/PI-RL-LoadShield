from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CONTROLLER_ORDER = [
    "Collective Pitch",
    "IBPC without Preview",
    "MPC Load Control",
    "Model-free DRL",
    "PI-RL LoadShield",
]

CONTROLLER_LABELS = {
    "Collective Pitch": "Collective pitch",
    "IBPC without Preview": "IBPC without preview",
    "MPC Load Control": "MPC load control",
    "Model-free DRL": "Model-free DRL",
    "PI-RL LoadShield": "PI-RL LoadShield",
}

CONTROLLER_COLORS = {
    "Collective Pitch": "#6f6f6f",
    "IBPC without Preview": "#d95f02",
    "MPC Load Control": "#1b9e77",
    "Model-free DRL": "#7570b3",
    "PI-RL LoadShield": "#0057a8",
}

ABLATION_ORDER = [
    "Full PI-RL LoadShield",
    "Without LiDAR preview",
    "Without physics penalty",
    "Without safety projection",
    "Without active yaw",
    "Without IBPC",
]

ABLATION_COLORS = {
    "Full PI-RL LoadShield": "#1b9e77",
    "Without LiDAR preview": "#d95f02",
    "Without physics penalty": "#377eb8",
    "Without safety projection": "#e41a1c",
    "Without active yaw": "#984ea3",
    "Without IBPC": "#8c564b",
}

SEVERITY_ORDER = ["Moderate", "High", "Severe", "Extreme"]
SEVERITY_INDEX = {"Moderate": 1, "High": 2, "Severe": 3, "Extreme": 4}


class InputBundle(Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]):
    pass


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def ensure_output_dir(root: Path) -> Path:
    out = root / "outputs"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    return out


def _candidate_csv_paths(filename: str, root: Path) -> List[Path]:
    cwd = Path.cwd()
    return [
        root / "data" / filename,
        root / filename,
        cwd / "data" / filename,
        cwd / filename,
    ]


def _candidate_zip_paths(root: Path) -> List[Path]:
    cwd = Path.cwd()
    return [
        root / "supplementary_materials.zip",
        root / "loadshield_remaining_open_source_files_fresh.zip",
        root / "loadshield_remaining_open_source_files.zip",
        cwd / "supplementary_materials.zip",
        cwd / "loadshield_remaining_open_source_files_fresh.zip",
        cwd / "loadshield_remaining_open_source_files.zip",
    ]


def read_csv_from_sources(filename: str, root: Path, required: bool = True) -> Optional[pd.DataFrame]:
    """Load an input CSV from data/, repository root, current directory, or a supplementary ZIP.

    This function only loads existing files. It does not create fallback values or inject paper
    target numbers. Optional files return None when unavailable.
    """
    for path in _candidate_csv_paths(filename, root):
        if path.exists():
            return pd.read_csv(path)

    for zip_path in _candidate_zip_paths(root):
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as archive:
            member = None
            for name in archive.namelist():
                if name == filename or name.endswith("/" + filename):
                    member = name
                    break
            if member:
                with archive.open(member) as fh:
                    return pd.read_csv(io.BytesIO(fh.read()))

    if required:
        raise FileNotFoundError(
            f"Missing {filename}. Place it in ./data/, beside the scripts, or inside the supplementary ZIP."
        )
    return None


def read_inputs(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    ts = read_csv_from_sources("loadshield_timeseries_dataset.csv", root, required=True)
    summary = read_csv_from_sources("episode_summary_metrics.csv", root, required=True)
    scenarios = read_csv_from_sources("scenario_config.csv", root, required=True)
    ablation = read_csv_from_sources("ablation_results.csv", root, required=False)
    robustness = read_csv_from_sources("robustness_grid_values.csv", root, required=False)
    assert ts is not None and summary is not None and scenarios is not None
    return ts, summary, scenarios, ablation, robustness


def require_columns(df: pd.DataFrame, required: Sequence[str], source_name: str) -> List[str]:
    missing = [col for col in required if col not in df.columns]
    if missing:
        warn(f"{source_name} is missing required columns: {missing}")
    return missing


def sort_controllers(df: pd.DataFrame, controller_col: str = "controller_type") -> pd.DataFrame:
    order = {name: idx for idx, name in enumerate(CONTROLLER_ORDER)}
    return (
        df.assign(_order=df[controller_col].map(order).fillna(999))
        .sort_values("_order")
        .drop(columns="_order")
    )


def mean_by_controller(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Mtb_peak_reduction_pct",
        "Mbr_rms_reduction_pct",
        "Tm_peak_reduction_pct",
        "Gamma_br_reduction_pct",
        "DEL_reduction_pct",
        "eP_RMS",
        "N_viol_applied",
        "an_rms_reduction_pct",
        "A_beta",
        "A_psi",
        "S_a",
        "L_inf_ms",
    ]
    available = [m for m in metrics if m in summary.columns]
    missing = [m for m in metrics if m not in summary.columns]
    if missing:
        warn(f"episode_summary_metrics.csv does not contain optional metrics: {missing}")
    grouped = summary.groupby("controller_type", as_index=False)[available].mean(numeric_only=True)
    return sort_controllers(grouped)


def build_table_controller_performance(summary: pd.DataFrame) -> pd.DataFrame:
    required = [
        "controller_type",
        "Mtb_peak_reduction_pct",
        "Mbr_rms_reduction_pct",
        "Tm_peak_reduction_pct",
        "Gamma_br_reduction_pct",
        "DEL_reduction_pct",
        "eP_RMS",
        "N_viol_applied",
    ]
    if require_columns(summary, required, "episode_summary_metrics.csv"):
        raise ValueError("Cannot build controller performance table because required columns are missing.")

    avg = mean_by_controller(summary)
    rows = []
    for _, row in avg.iterrows():
        ctrl = row["controller_type"]
        if ctrl == "Collective Pitch":
            values = ["Reference"] * 5
        else:
            values = [
                f"{row['Mtb_peak_reduction_pct']:.1f}%",
                f"{row['Mbr_rms_reduction_pct']:.1f}%",
                f"{row['Tm_peak_reduction_pct']:.1f}%",
                f"{row['Gamma_br_reduction_pct']:.1f}%",
                f"{row['DEL_reduction_pct']:.1f}%",
            ]
        rows.append(
            {
                "Controller": CONTROLLER_LABELS.get(ctrl, ctrl),
                "M_tb^peak ↓": values[0],
                "M_br^RMS ↓": values[1],
                "T_m^peak ↓": values[2],
                "Gamma_br ↓": values[3],
                "DEL ↓": values[4],
                "e_P^RMS": f"{row['eP_RMS']:.3f}",
                "N_viol^applied": str(int(round(row["N_viol_applied"]))),
            }
        )
    return pd.DataFrame(rows)


def build_table_feasibility(summary: pd.DataFrame) -> pd.DataFrame:
    required = ["controller_type", "an_rms_reduction_pct", "A_beta", "A_psi", "S_a", "L_inf_ms"]
    if require_columns(summary, required, "episode_summary_metrics.csv"):
        raise ValueError("Cannot build feasibility table because required columns are missing.")

    avg = mean_by_controller(summary)
    rows = []
    for _, row in avg.iterrows():
        ctrl = row["controller_type"]
        rows.append(
            {
                "Controller": CONTROLLER_LABELS.get(ctrl, ctrl),
                "a_n^RMS ↓": "Reference" if ctrl == "Collective Pitch" else f"{row['an_rms_reduction_pct']:.1f}%",
                "A_beta*": f"{row['A_beta']:.2f}",
                "A_psi*": f"{row['A_psi']:.2f}",
                "S_a": f"{row['S_a']:.4f}",
                "L_inf": f"{row['L_inf_ms']:.1f} ms",
            }
        )
    return pd.DataFrame(rows)


def derive_ablation_table(summary: pd.DataFrame, ablation: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return ablation values from ablation_results.csv, or matching rows in summary.

    No synthetic offsets or manuscript target values are used. If the rows/columns are missing,
    the function returns None and the corresponding figure is skipped.
    """
    source = ablation if ablation is not None else summary
    if source is None or source.empty:
        warn("Ablation source data is unavailable.")
        return None

    variant_col = "variant" if "variant" in source.columns else "controller_type"
    if variant_col not in source.columns:
        warn("Ablation data requires either 'variant' or 'controller_type'.")
        return None

    metric_candidates: Dict[str, List[str]] = {
        "M_tb_peak": ["M_tb_peak", "Mtb_peak_reduction_pct", "M_tb_peak_reduction_pct"],
        "Gamma_br": ["Gamma_br", "Gamma_br_reduction_pct"],
        "T_m_peak": ["T_m_peak", "Tm_peak_reduction_pct", "T_m_peak_reduction_pct"],
        "DEL": ["DEL", "DEL_reduction_pct"],
        "N_viol": ["N_viol", "N_viol_applied", "N_viol_pre"],
    }

    selected = source[source[variant_col].isin(ABLATION_ORDER)].copy()
    missing_variants = [v for v in ABLATION_ORDER if v not in set(selected[variant_col].dropna())]
    if missing_variants:
        warn("Ablation variants missing: " + ", ".join(missing_variants))
        return None

    resolved: Dict[str, str] = {}
    for output_name, candidates in metric_candidates.items():
        found = next((c for c in candidates if c in selected.columns), None)
        if found is None:
            warn(f"Ablation metric {output_name} missing. Expected one of {candidates}.")
            return None
        resolved[output_name] = found

    grouped = selected.groupby(variant_col, as_index=False)[list(resolved.values())].mean(numeric_only=True)
    grouped = grouped.rename(columns={variant_col: "Variant", **{v: k for k, v in resolved.items()}})
    order = {name: idx for idx, name in enumerate(ABLATION_ORDER)}
    grouped = grouped.assign(_order=grouped["Variant"].map(order)).sort_values("_order").drop(columns="_order")
    grouped["N_viol"] = grouped["N_viol"].round().astype(int)
    return grouped


def robustness_grid(summary: pd.DataFrame, robustness: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return robustness grid from recorded robustness data or summary columns."""
    source = robustness if robustness is not None else summary
    if source is None or source.empty:
        warn("Robustness source data is unavailable.")
        return None

    if require_columns(source, ["storm_severity", "communication_delay_ms"], "robustness source data"):
        return None

    candidates: Dict[str, List[str]] = {
        "Mtb_peak_normalized": ["Mtb_peak_normalized", "Mtb_norm_peak", "M_tb_norm_peak"],
        "N_viol_pre": ["N_viol_pre", "N_viol_pre_cumulative_final", "raw_action_violation_count"],
        "N_viol_applied": ["N_viol_applied", "N_viol_applied_cumulative_final", "applied_action_violation_count"],
        "normalized_safety_margin": ["normalized_safety_margin", "min_mooring_safety_margin", "mooring_safety_margin_min"],
    }
    resolved: Dict[str, str] = {}
    for output_name, options in candidates.items():
        found = next((c for c in options if c in source.columns), None)
        if found is None:
            warn(f"Robustness metric {output_name} missing. Expected one of {options}.")
            return None
        resolved[output_name] = found

    work = source.copy()
    if "controller_type" in work.columns and "PI-RL LoadShield" in set(work["controller_type"].dropna()):
        work = work[work["controller_type"] == "PI-RL LoadShield"].copy()
    if "storm_severity_index" not in work.columns:
        work["storm_severity_index"] = work["storm_severity"].map(SEVERITY_INDEX)
    work = work.dropna(subset=["storm_severity_index"])

    grouped = work.groupby(
        ["storm_severity", "storm_severity_index", "communication_delay_ms"], as_index=False
    )[list(resolved.values())].mean(numeric_only=True)
    grouped = grouped.rename(columns={v: k for k, v in resolved.items()})
    grouped["storm_severity_index"] = grouped["storm_severity_index"].astype(int)
    grouped["communication_delay_ms"] = grouped["communication_delay_ms"].astype(float)
    return grouped


def save_result_tables(summary: pd.DataFrame, ablation: Optional[pd.DataFrame], robustness: Optional[pd.DataFrame], out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    table7 = build_table_controller_performance(summary)
    table8 = build_table_feasibility(summary)
    table7.to_csv(out_dir / "tables" / "table7_controller_performance.csv", index=False)
    table8.to_csv(out_dir / "tables" / "table8_feasibility_metrics.csv", index=False)

    abl = derive_ablation_table(summary, ablation)
    if abl is not None:
        abl.to_csv(out_dir / "tables" / "fig6_ablation_values.csv", index=False)

    rob = robustness_grid(summary, robustness)
    if rob is not None:
        rob.to_csv(out_dir / "tables" / "fig7_robustness_grid.csv", index=False)

    return table7, table8
