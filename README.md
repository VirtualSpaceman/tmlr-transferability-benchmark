# TMLR Transferability Benchmark

Benchmark and analysis pipeline for transferability scorers, including score computation, bootstrap-based correlation summaries, and figure generation used in the TMLR study.

## What this repository does

This project evaluates transferability scorers against downstream test performance across datasets/models, then summarizes and visualizes ranking quality (mainly Kendall tau and weighted Kendall tau).

Typical flow:
1. Read transfer scores from CSV files in `inputs/`.
2. Run per-scorer computations with Nextflow (`experiments/correlations_compute.nf`).
3. Summarize bootstrap outputs (`experiments/correlations_summarize.nf`).
4. Generate figures/tables from summary JSON files (`analysis/plots/*` and notebooks).

## Repository structure

### Top-level folders

- `analysis/`: Python modules and notebooks for computation, filtering, merging, and plotting.
- `experiments/`: Nextflow workflows, runtime config, and generated workflow outputs.
- `inputs/`: Source CSV/JSON datasets used by the workflows and plotting scripts.
- `outputs/`: Final/exported plots and consolidated JSON outputs used in the paper analysis.

## Main files and their roles

### `experiments/` (workflow layer)

- `experiments/nextflow.config`: Local execution profile (executor, CPUs, memory, queue size, submit rate).
- `experiments/correlations_compute.nf`: Main compute pipeline. Runs one task per scorer and writes `correlations_<scorer>.json` and logs.
- `experiments/correlations_summarize.nf`: Summarization pipeline. Converts raw per-scorer outputs into `summ_*.json` diagnostics/statistics used by plots.
- `experiments/count_callibration_tuples.nf`: Auxiliary workflow to inspect tuple counts for prediction experiments (explicitly marked as non-official).
- `experiments/prepare_environment.sh`: Creates/activates Conda env `stan` and installs required Python packages.
- `experiments/merge_csv.sh`: Small shell helper that concatenates two CSV files while keeping one header.
- `experiments/results_sota/`: Published output directory for per-scorer raw computation JSON/log files.
- `experiments/summary_sota/`: Published output directory for summarized JSON files.

### `analysis/` (computation and utilities)

- `analysis/correlations_compute.py`: Per-scorer core computation. Loads CSV, computes tau/wtau (dataset-wise and overall), bootstraps samples, and writes JSON payloads.
- `analysis/correlations_summarize.py`: Reads compute JSON files and derives means, HDIs, and sampled summary statistics.
- `analysis/utils.py`: Shared library (bootstrap utilities, categorical encoding, normalization helpers, Kendall implementations, plotting helpers).
- `analysis/filter_csv.py`: Generic CSV filter CLI (`eval`-based condition + optional summary print).
- `analysis/filter_data.py`: CSV filter utility with behavior equivalent to `filter_csv.py` (legacy/duplicated helper).
- `analysis/filter_json.py`: JSON key-path selector and formatter (supports key extraction, prefix wrapping, indentation, sorting).
- `analysis/merge_json.py`: Recursive JSON merge utility with optional collision override (`--force`).
- `analysis/print_callibration_tuple_count.py`: Utility script to join ridgeline configuration JSON with tuple-count CSV and print merged counts.
- `analysis/test_kendall.py`: Local script for validating Kendall-related behavior/experiments.

### `analysis/plots/` (figure generation)

- `analysis/plots/main_scatterplot.py`: Main scatterplot generator with dataset/model legends and optional bootstrap annotations.
- `analysis/plots/main_scatterplot_taus.py`: Variant focused on tau/wtau-driven scatter visualization.
- `analysis/plots/main_scatterplot_single_color.py`: Scatter variant with simplified color styling.
- `analysis/plots/linear_main_scatterplot.py`: Scatterplot generation for linear-ablation experiment outputs.
- `analysis/plots/ablations_ridgelineplots.py`: Ridgeline plots for ablation summaries (KDE/hist/sample modes).
- `analysis/plots/mod_ridgelineplots.py`, `analysis/plots/ridgelineplots.py`, `analysis/plots/single_ridgelineplots.py`, `analysis/plots/ridgelineplots_inverted.py`: Additional ridgeline variants and layout-specific renderers.
- `analysis/plots/*.json` (for ridgelines): Plot layout/config descriptors that define rows/columns/prefix combinations.

### Analysis notebooks and experiment scripts

- `analysis/new_plots_tmlr_ensemble.py`: Script-style exploratory workflow for linear pooling/SVR/least-squares combinations and ablation CSV generation.
- `analysis/*.ipynb` notebooks (`prepare_ablations_csv_*`, `save_linear_ablations.ipynb`, `plots_tmlr_correlations.ipynb`, etc.): Interactive analysis and plotting pipelines used for exploratory and publication-oriented figures.

### Data folders

- `inputs/transf_scores.csv`: Primary benchmark input used by default workflows.
- `inputs/transf_scores_frozen.csv` and `inputs/frozen_*`: Frozen snapshots for reproducible reruns.
- `inputs/linear_ablations_*.csv`, `inputs/tmlr_linear_ablations_combinations.csv`: Inputs for linear ablation analyses.
- `outputs/*.pdf`: Final rendered visualizations.
- `outputs/*.json`: Aggregated/scored outputs for downstream analysis.

## Environment setup

From repository root:

```bash
cd experiments
bash prepare_environment.sh
```

This creates/updates Conda env `stan` and installs:
- arviz
- matplotlib
- numpy
- pandas
- pystan
- scikit-learn
- scipy

You also need Nextflow available in your shell.

## Running the main workflows

From `experiments/`:

```bash
# 1) Compute per-scorer correlations
nextflow run correlations_compute.nf

# 2) Summarize computed results
nextflow run correlations_summarize.nf
```

Default outputs:
- raw scorer outputs: `experiments/results_sota/`
- summarized outputs: `experiments/summary_sota/`

## Running plotting scripts

Many plotting scripts are Python CLIs under `analysis/plots/`.

Example (from repository root):

```bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
python -m analysis.plots.main_scatterplot \
  --input_file inputs/transf_scores.csv \
  --input_boots outputs/scorers_with_btb_bootstrapped_grouped.json \
  --output_file outputs/sota_scatterplot_wtau.pdf
```

For ridgeline plots, use `analysis/plots/ablations_ridgelineplots.py` and point `--input_dirs` to one or more summary folders.

## Notes and troubleshooting

- Nextflow pipelines in this repository activate Conda env `stan` internally.
- If a process fails with "missing output file", inspect the corresponding `.log` file in the task/work directory; Python errors can be hidden by shell piping.
- Some scripts are exploratory and may contain dataset/scorer subsets commented in/out for specific experiments.
