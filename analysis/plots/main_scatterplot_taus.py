import argparse
import json

import numpy as np
import pandas as pd

from ..utils import bootstrap, ci_from_bootstrap, grouped_weighted_kendall_tau
from .main_scatterplot import DATASET_CLASSES_N, DATASET_TO_COLOR, TRANSF_METRICS_TO_MARKERS


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Scatterplots with classical regressions.')

    # Add the arguments
    parser.add_argument('--input_file', required=True, help='Path to the input .csv file with the experimental data.')
    parser.add_argument('--output_file', required=True, help="Path to the output json file with bootstrapped grouped"
                        "Kendall's tau statistics.")
    args = parser.parse_args()

    INPUT_PATH = args.input_file
    OUTPUT_FILE = args.output_file
    ALPHA = 0.05

    # Sample data
    df = pd.read_csv(INPUT_PATH)

    unique_datasets = df['dataset'].unique()
    if not set(unique_datasets).issubset(set(DATASET_CLASSES_N.keys())):
        print(f'Unique datasets: {unique_datasets}')
        print(f'Available datasets: {DATASET_CLASSES_N.keys()}')
        raise ValueError('Dataset names do not match.')

    transf_metrics = sorted(df['transf_metric'].unique())

    transf_metrics_taus = {}

    for transf_metric in TRANSF_METRICS_TO_MARKERS:
        if transf_metric not in transf_metrics:
            continue
        print(f'Processing {transf_metric}...')

        all_x = []
        all_y = []
        all_d = []
        for d, dataset in enumerate(DATASET_TO_COLOR):
            if dataset not in unique_datasets:
                continue
            subset = df[(df['dataset'] == dataset) & (df['transf_metric'] == transf_metric)]
            x = subset['transf_score'].values
            y = subset['test_score'].values
            all_x.extend(x)
            all_y.extend(y)
            all_d.extend([d]*len(x))

        tau_maxll = grouped_weighted_kendall_tau(all_x, all_y, all_d)
        print(f'... tau = {tau_maxll:.4f}')
        tau_boots = bootstrap((all_x, all_y, all_d), grouped_weighted_kendall_tau, n_bootstraps=1000)
        tau_ci = ci_from_bootstrap(tau_boots, alpha=ALPHA)

        # Store the results
        print(f'... mean = [{np.mean(tau_boots):.4f}]')
        print(f'... CI = [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}]')
        # TODO: this is indexed as 'tau' but is the 'wtau'
        transf_metrics_taus[transf_metric] = {'tau': tau_maxll, 'ci': tau_ci, 'boots': tau_boots}

    print('Saving...')
    with open(OUTPUT_FILE, 'wt', encoding='utf-8') as f:
        json.dump(transf_metrics_taus, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
