# PI-RL LoadShield: Reproducibility Package for Storm-Time Load Mitigation in Floating Offshore Wind Turbines

This repository contains the dataset files, Python reproduction code, generated figures, and code-output snapshots for the study:

**PI-RL LoadShield: Physics-Integrated Reinforcement Learning for Storm-Time Load Mitigation in Floating Offshore Wind Turbines**

The repository is prepared to support open-data release, reviewer verification, GitHub sharing, and later Zenodo archiving.

---

## 1. Project Overview

Floating offshore wind turbines can harvest strong deep-sea winds, but their floating platforms are vulnerable to coupled wind–wave loading. During storm conditions, rotor thrust, platform pitch, yaw misalignment, tower-base moment, blade-root load, nacelle acceleration, and mooring tension can rise together within a short time.

**PI-RL LoadShield** is a physics-integrated reinforcement learning framework designed as a storm-response support layer for floating offshore wind turbines. The framework combines:

- 5G-LiDAR wind preview,
- structural state estimation,
- Proximal Policy Optimization (PPO)-based control,
- independent blade pitch control,
- active yaw correction,
- physics-aware reward shaping,
- deterministic safety projection,
- digital-twin-based validation.

The released files allow readers to inspect the input datasets, run the reproduction script, generate tables and figures, and compare the outputs with the values reported in the manuscript.

---

## 2. Repository Structure

The repository is organized using only the files included in this release.

```text
PI-RL-LoadShield/
│
├── README.md
├── requirements.txt
├── reproduce_results.py
├── loadshield_data.py
├── loadshield_figures.py
│
├── data/
│   ├── loadshield_timeseries_dataset.csv
│   ├── episode_summary_metrics.csv
│   ├── scenario_config.csv
│   └── loadshield_column_description.txt
│
├── figures/
│   ├── figure4_time_domain_response.png
│   ├── figure5_feasibility_dashboard.png
│   ├── figure6_unified_ablation_plot.png
│   └── figure7_robustness_heatmaps.png
│
└── snapshots/
    ├── 01_terminal_success_output.png
    ├── 02_output_folder_tree.png
    ├── 03_table7_csv_output.png
    ├── 04_table8_csv_output.png
    ├── 05_fig6_ablation_csv_output.png
    └── 06_fig7_robustness_grid_csv_output.png
```

---

## 3. File Descriptions

### Root-level code files

| File | Description |
|---|---|
| `reproduce_results.py` | Main script used to reproduce the result tables and figures from the released CSV files. |
| `loadshield_data.py` | Handles dataset loading, required-column checks, controller-level aggregation, table generation, ablation-value extraction, and robustness-grid construction. |
| `loadshield_figures.py` | Generates the manuscript-style figures from the processed datasets. |
| `requirements.txt` | Lists the Python packages required to run the reproduction code. |
| `README.md` | Explains the repository structure, usage, authorship, license, and citation information. |

### Data files

| File | Description |
|---|---|
| `data/loadshield_timeseries_dataset.csv` | Main time-series dataset containing environmental inputs, LiDAR-preview variables, controller states, actuator commands, and structural-response variables. |
| `data/episode_summary_metrics.csv` | Episode-level controller performance summary used for reproducing controller comparison metrics, feasibility metrics, and figure-level summaries. |
| `data/scenario_config.csv` | Scenario configuration file containing storm-event settings, severity information, delay levels, and validation conditions. |
| `data/loadshield_column_description.txt` | Column description file explaining variables, units, and dataset meaning. |

### Figure files

| File | Description |
|---|---|
| `figures/figure4_time_domain_response.png` | Time-domain response under a combined gust–wave event. |
| `figures/figure5_feasibility_dashboard.png` | Fatigue–power–actuator feasibility dashboard. |
| `figures/figure6_unified_ablation_plot.png` | Unified ablation-performance plot of PI-RL LoadShield. |
| `figures/figure7_robustness_heatmaps.png` | Robustness analysis under communication delay and storm severity. |

### Snapshot files

| File | Description |
|---|---|
| `snapshots/01_terminal_success_output.png` | Terminal output after successful execution of the reproduction script. |
| `snapshots/02_output_folder_tree.png` | Output-folder structure created after running the code. |
| `snapshots/03_table7_csv_output.png` | Screenshot of generated Table 7 CSV output. |
| `snapshots/04_table8_csv_output.png` | Screenshot of generated Table 8 CSV output. |
| `snapshots/05_fig6_ablation_csv_output.png` | Screenshot of generated ablation values used for Figure 6. |
| `snapshots/06_fig7_robustness_grid_csv_output.png` | Screenshot of generated robustness-grid values used for Figure 7. |

---

## 4. Installation

The code was prepared for Python 3.9 or later.

First, clone or download this repository.

Then install the required packages:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
numpy
pandas
matplotlib
```

---

## 5. How to Reproduce the Results

From the repository root, run:

```bash
python reproduce_results.py --root .
```

The script reads the input files from the `data/` folder and writes generated outputs to:

```text
outputs/tables/
outputs/figures/
```

To generate only the tables and skip figures, run:

```bash
python reproduce_results.py --root . --skip-figures
```

---

## 6. Expected Generated Output

After successful execution, the script generates an `outputs/` folder similar to this:

```text
outputs/
├── figures/
│   ├── figure4_time_domain_response.png
│   ├── figure4_time_domain_response.pdf
│   ├── figure5_feasibility_dashboard.png
│   ├── figure5_feasibility_dashboard.pdf
│   ├── figure6_unified_ablation_plot.png
│   ├── figure6_unified_ablation_plot.pdf
│   ├── figure7_robustness_heatmaps.png
│   └── figure7_robustness_heatmaps.pdf
│
└── tables/
    ├── table7_controller_performance.csv
    ├── table8_feasibility_metrics.csv
    ├── fig6_ablation_values.csv
    └── fig7_robustness_grid.csv
```

The `snapshots/` folder contains screenshots of a successful code run and the generated output files.

---

## 7. Main Reproduced Tables

### Table 7: Quantitative comparison of controller performance

The script reproduces the following controller-level metrics:

- peak tower-base moment reduction,
- blade-root RMS load reduction,
- peak mooring tension reduction,
- blade-wise imbalance reduction,
- fatigue damage equivalent load reduction,
- RMS power error,
- applied safety-violation count.

### Table 8: Power, actuator, and real-time feasibility metrics

The script reproduces:

- nacelle acceleration RMS reduction,
- normalized cumulative pitch travel,
- normalized cumulative yaw travel,
- action smoothness index,
- inference latency.

---

## 8. Main Reproduced Figures

The code and released files support the following figure outputs:

| Figure | Description |
|---|---|
| Figure 4 | Time-domain response under a combined gust–wave event. |
| Figure 5 | Fatigue–power–actuator feasibility dashboard. |
| Figure 6 | Unified ablation-performance plot showing the effect of removing key framework modules. |
| Figure 7 | Robustness heatmaps under communication delay and storm severity. |

---

## 9. Dataset Notes

The dataset is organized to support reproducibility of the reported manuscript results. It includes:

- environmental and storm-event variables,
- LiDAR-equivalent preview features,
- controller states,
- pitch and yaw command outputs,
- floating-turbine structural-response variables,
- episode-level performance metrics,
- scenario-level validation settings.

The dataset is intended for:

- reproducibility checking,
- controller benchmarking,
- floating offshore wind turbine load-mitigation research,
- digital-twin-based controller evaluation,
- physics-integrated reinforcement learning studies.

The dataset should **not** be interpreted as direct field measurements from an operational offshore wind turbine. It is intended for research and reproducibility purposes.

---

## 10. Reproducibility Statement

All generated tables and figures are produced from the released CSV files using the Python scripts provided in this repository.

The workflow is:

```text
data CSV files
        ↓
reproduce_results.py
        ↓
loadshield_data.py + loadshield_figures.py
        ↓
outputs/tables/ and outputs/figures/
```

The code reads the input CSV files, validates required columns, aggregates controller metrics, and saves the reproduced tables and figures.

---

## 11. Authors and Affiliations

**Shaoyu Wang**<sup>1,a</sup>, **Xiaokai Zhou**<sup>2,b</sup>, **Weiyue Feng**<sup>3,c</sup>, **Ye Wang**<sup>4,d</sup>, and **Fangang Zeng**<sup>5,e,*</sup>

<sup>1</sup> School of Ship and Marine Engineering, Dalian Maritime University, Dalian 123456, Liaoning, China  
<sup>2</sup> Department of Energy and Power Engineering, Tsinghua University, Beijing 100084, China  
<sup>3</sup> School of Electrical Engineering, Beijing Jiaotong University, Beijing 100044, China  
<sup>4</sup> College of Chemistry and Chemical Engineering, Shaanxi University of Science & Technology, Xi'an 123456, Shaanxi Province, China  
<sup>5</sup> School of Chemistry and Life Resources, Renmin University of China, Beijing 100872, China  

### Emails

- <sup>a</sup> 13503530012@163.com  
- <sup>b</sup> zhouxk25@mails.tsinghua.edu.cn  
- <sup>c</sup> 24241177@bjtu.edu.cn  
- <sup>d</sup> 13935350968@163.com  
- <sup>e</sup> mprema835@gmail.com  

\* Corresponding author: **Fangang Zeng**  
Email: mprema835@gmail.com

---

## 12. Suggested Citation

If you use this repository, dataset, or code, please cite the associated manuscript and the archived version of this repository.

Suggested citation:

```text
Wang, S.; Zhou, X.; Feng, W.; Wang, Y.; Zeng, F. PI-RL LoadShield: Physics-Integrated Reinforcement Learning for Storm-Time Load Mitigation in Floating Offshore Wind Turbines. [Journal details to be added].
```

After Zenodo archiving, the DOI will be added here.

---

## 13. License

Unless otherwise stated, the repository is released for academic and research use.

Suggested licensing arrangement:

- **Code files**: MIT License  
- **Dataset, figures, tables, and documentation**: Creative Commons Attribution 4.0 International License (CC BY 4.0)

Users are allowed to use, modify, and redistribute the code and data for research purposes, provided that appropriate credit is given to the authors and the associated manuscript is cited.

---

## 14. Disclaimer

This repository is provided for research and reproducibility purposes. The released dataset and code are not intended for turbine certification, commercial deployment, safety approval, or field-level offshore controller operation without further validation.

The authors provide the material as-is and do not guarantee fitness for operational wind turbine control.

---

## 15. Contact

For questions regarding the dataset, code, or reproducibility package, please contact:

**Fangang Zeng**  
School of Chemistry and Life Resources, Renmin University of China, Beijing 100872, China  
Email: mprema835@gmail.com

---

## 16. Repository Status

This repository is prepared for GitHub release and later Zenodo archiving. The final Zenodo DOI will be added after archival upload.
