# PI-RL LoadShield FOWT Digital-Twin Dataset

This repository contains supplementary open data and reproducibility material for the study:

**Autonomous Physics-Integrated Reinforcement Learning for Real-Time Structural Load Mitigation in Floating Offshore Wind Turbines via IoT-Driven Adaptive Pitch and Yaw Mechanization**

The material supports the PI-RL LoadShield framework, a storm-time structural load-shedding approach for floating offshore wind turbines.

## Important data statement

This dataset is generated from a digital-twin simulation workflow aligned with floating offshore wind turbine response variables. It does not contain field measurements from an operational offshore turbine.

## Included files

- `loadshield_timeseries_dataset.csv`: main time-series dataset.
- `episode_summary_metrics.csv`: episode-level metrics.
- `scenario_config.csv`: scenario definitions.
- `ablation_results.csv`: ablation-study values.
- `robustness_grid_values.csv`: delay-severity robustness grid.
- `loadshield_column_description.txt`: column descriptions.
- `reproduce_results.py`: script for regenerating tables and figures.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python reproduce_results.py
```

Outputs are written to:

```text
outputs/tables/
outputs/figures/
```

## Scope

The release is intended for reproducibility, benchmarking, and review transparency. It should not be interpreted as field certification of a deployed offshore turbine controller.
