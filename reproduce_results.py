#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loadshield_data import ensure_output_dir, read_inputs, save_result_tables
from loadshield_figures import generate_all_figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce PI-RL LoadShield result tables and figures from released CSV files."
    )
    parser.add_argument("--root", type=str, default=".", help="Repository root containing data/ or CSV files.")
    parser.add_argument("--skip-figures", action="store_true", help="Generate tables only.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = ensure_output_dir(root)

    try:
        ts, summary, scenarios, ablation, robustness = read_inputs(root)
        table7, table8 = save_result_tables(summary, ablation, robustness, out_dir)
        if not args.skip_figures:
            generate_all_figures(ts, summary, ablation, robustness, out_dir)
    except Exception as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        return 1

    print("\nTable 7. Quantitative comparison of controller performance")
    print(table7.to_string(index=False))
    print("\nTable 8. Power, actuator, and real-time feasibility metrics")
    print(table8.to_string(index=False))
    print("\nGenerated outputs")
    print(f"  Tables : {out_dir / 'tables'}")
    print(f"  Figures: {out_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
