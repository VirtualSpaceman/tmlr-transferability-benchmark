'''
Classical regression summary from simulations.
'''
import argparse
from itertools import chain
import json

import arviz as az
import numpy as np


HDI_TAIL_PROB = 0.05

# Create the parser
parser = argparse.ArgumentParser(description='Bayesian R-Squared summary from simulations.')

# Add the arguments
parser.add_argument('--input_file', required=True, help='Path to the input .json file with the visualization data.')
parser.add_argument('--output_file', help='Optional output .json for the results.')
parser.add_argument('--hdi_tail_prob', type=float, default=HDI_TAIL_PROB,
                    help=f'Tail probability of the highest density intervals. Defaults to {HDI_TAIL_PROB}.')

# Parse the arguments
args = parser.parse_args()
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file
HDI_TAIL_PROB = args.hdi_tail_prob

print('Preparing data...')
# Read the visualization data
with open(INPUT_FILE, 'rt', encoding='utf-8') as f:
    data = json.load(f)

summary = {}

z_score = np.asarray(data['input']['z_score'])
z_metric = np.asarray(data['input']['z_metric'])
metric = np.asarray(data['input']['metric'])
i_dataset = np.asarray(data['input']['dataset'])
i_datasets = sorted(set(data['input']['dataset']))
dataset_names = {int(k): v for k, v in data['translation']['dataset'].items()}  # json keys are always strings
boots_data = data['output']

for suffix in chain(i_datasets, ['OVERALL']):
    suffix_out = suffix if suffix == 'OVERALL' else dataset_names[suffix]

    # Kendall's tau
    tau_boots = np.asarray( boots_data[f'bootstrapped_tau_{suffix}'])
    tau_mean = tau_boots.mean()
    tau_maxll = boots_data[f'tau_{suffix}']

    # Weighted Kendall's tau
    wtau_boots = np.asarray( boots_data[f'bootstrapped_wtau_{suffix}'])
    wtau_mean = wtau_boots.mean()
    wtau_maxll = boots_data[f'wtau_{suffix}']

    summary[suffix_out] = {
        'tau': {
            'maxll': tau_maxll,
            'mean': tau_mean,
            'mean-maxll': tau_mean - tau_maxll,
            'HDI': az.hdi(tau_boots, hdi_prob=1.-HDI_TAIL_PROB).tolist(),
            'samples': tau_boots.tolist(),
        },
        'wtau': {
            'maxll': wtau_maxll,
            'mean': wtau_mean,
            'mean-maxll': wtau_mean - wtau_maxll,
            'HDI': az.hdi(wtau_boots, hdi_prob=1.-HDI_TAIL_PROB).tolist(),
            'samples': wtau_boots.tolist(),
        },
    }

if OUTPUT_FILE is not None:
    print(f'Writing the results to {OUTPUT_FILE}...')
    with open(OUTPUT_FILE, 'wt', encoding='utf-8') as f:
        json.dump(summary, f)


def print_summary(s, prefix=''):
    for k, v in s.items():
        if isinstance(v, dict):
            print_summary(v, prefix=f'{prefix}{k} > ')
        elif isinstance(v, list):
            print(f'{prefix}{k}:', ', '.join(map(str, v)))
        else:
            print(f'{prefix}{k}: {v}')


print('\nSummary:')
print_summary(summary)
