'''
Classical regression model for a single transfer metric.
'''
import argparse
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, weightedtau

from .utils import ToListEncoder, bootstrap, encode_categorical, grouped_weighted_kendall_tau


def fit_regression_model(x, y, *, groups=None):

    def is_constant_array(arr):
        return np.all(arr == arr[0])

    def kendall_taus(x, y):
        if is_constant_array(x):
            if is_constant_array(y):
                # This should be an exceedingly rare situation in which we sampled the same value n times
                pass
            else:
                # If the scores are constant when the metric is not, we assign no correlation/predictive power to the
                # sample we handle this explicitly because kendalltau/weightedtau returns NaN in this case
                return 0, 0
        tau, _ = kendalltau(x, y)
        if np.isnan(tau) or np.isinf(tau):
            print('Warning: Nonfinite tau', tau, 'for', x, y)
        wtau, _ = weightedtau(x, y)
        return tau, wtau

    def grouped_kendall_taus(x, y, g):
        if is_constant_array(x):
            if is_constant_array(y):
                pass
            else:
                return 0, 0
        tau = grouped_weighted_kendall_tau(x, y, g, weighter=lambda r: 1)
        if np.isnan(tau) or np.isinf(tau):
            print('Warning: Nonfinite tau', tau, 'for', x, y)
        wtau = grouped_weighted_kendall_tau(x, y, g)
        return tau, wtau

    if groups is None:
        tau, wtau = kendall_taus(x, y)
        kendall = bootstrap((x, y), kendall_taus, n_bootstraps=BOOTSTRAP_ITERATIONS)
    else:
        tau, wtau = grouped_kendall_taus(x, y, groups)
        kendall = bootstrap((x, y, groups), grouped_kendall_taus, n_bootstraps=BOOTSTRAP_ITERATIONS)

    return dict(
            tau=tau,
            wtau=wtau,
            bootstrapped_tau=[k[0] for k in kendall],
            bootstrapped_wtau=[k[1] for k in kendall],
        )


def dict_suffix(d, suffix):
    return {k+suffix: v for k, v in d.items()}


if __name__ == '__main__':
    SIGNIFICANCE_LEVEL = 0.05
    SCRIPT_FILE = os.path.abspath(__file__)
    BOOTSTRAP_ITERATIONS = 1000 
    # Create the parser
    parser = argparse.ArgumentParser(description='Classical regression model for a single transfer metric.')

    # Add the arguments
    parser.add_argument('--scorer', required=True, help='Name of the transfer scorer to evaluate.')
    parser.add_argument('--input_file', required=True, help='Path to the input .csv file with the experimental data.')
    parser.add_argument('--output_file',  required=True, help='Output .json file with fit model.')
    parser.add_argument('--translation_file', help='Optional .json output with the encoding of categorical variables.')
    parser.add_argument('--significance_level', type=float, default=SIGNIFICANCE_LEVEL, help='Significance level for '
                        f'statistical tests, given by the tail probability. Defaults to {SIGNIFICANCE_LEVEL}.')

    # Parse the arguments
    args = parser.parse_args()
    SIGNIFICANCE_LEVEL = args.significance_level
    SCORER = args.scorer
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    TRANSLATION_FILE = args.translation_file

    print('Preparing data...')
    # Read the dataset
    df = pd.read_csv(INPUT_FILE)

    df = df[df['transf_metric'] == SCORER]

    # Z-normalize the test scores and the transfer scores per group of transfer metric and dataset
    # ... by default do nothing
    df['z_test_score'] = df['test_score']
    df['z_transf_score'] = df['transf_score']

    # Encode the categorical variables as integers
    translation = encode_categorical(df, columns=['model', 'dataset'])
    if TRANSLATION_FILE is not None:
        with open(TRANSLATION_FILE, 'wt', encoding='utf-8') as f:
            json.dump(translation, f, ensure_ascii=False, indent=4)

    # Prepare data for the bootstrapping
    n_datasets = df['i_dataset'].nunique()
    transfer_data = {
        'N': len(df),
        'D': n_datasets,
        'dataset': df['i_dataset'].tolist(),
        'z_metric': df['z_test_score'].tolist(),
        'z_score': df['z_transf_score'].tolist()
    }

    print('Fitting models...')
    results = {}
    outputs = {}
    for i_dataset in sorted(df['i_dataset'].unique()):
        # print('...dataset', i_dataset)
        score = df['z_transf_score'][df['i_dataset'] == i_dataset]
        metric = df['z_test_score'][df['i_dataset'] == i_dataset]
        pairs = sorted(zip(score, metric))
        score = np.array([p[0] for p in pairs])
        metric = np.array([p[1] for p in pairs])
        results_db = fit_regression_model(score, metric)
        results.update(dict_suffix(results_db, f'_{i_dataset}'))

    print('...overall')
    score = df['z_transf_score'].to_numpy()
    metric = df['z_test_score'].to_numpy()
    groups = df['i_dataset'].to_numpy()
    results_db = fit_regression_model(score, metric, groups=groups)
    results.update(dict_suffix(results_db, '_OVERALL'))

    # Saves the results
    # ... adds raw data to inputs before saving
    transfer_data['metric'] = df['test_score'].tolist()
    transfer_data['score'] = df['transf_score'].tolist()
    transfer_data['model'] = df['i_model'].tolist()
    output = {
        'input': transfer_data,
        'output': results,
        'translation': translation,
        'script': os.path.basename(SCRIPT_FILE),
        'args': vars(args),
        'timestamp': datetime.utcnow().isoformat(),
    }
    with open(OUTPUT_FILE, 'wt', encoding='utf-8') as f:
        json.dump(output, f, cls=ToListEncoder)

    # Print the summary
    print('\n\nSummary of the model:')
    for k, v in results.items():
        if k.startswith('bootstrapped_'):
            continue
        print(f'{k}: {v:.3f}')
