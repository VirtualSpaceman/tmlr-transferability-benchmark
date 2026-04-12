import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

from types import MappingProxyType as rodict
from ..utils import bootstrap, ci_from_bootstrap, grouped_weighted_kendall_tau

DATASET_CLASSES_N = rodict({
    'sun397':             397,
    'aircraft':           100,
    'caltech101':         102,
    'oxfordpets':         37,
    'flowers102':         102,
    'dtd':                47,
    'stanfordcars':       196,
    'voc2007':            20,
    'brain_tumor_kaggle': 4,
    'breakhis':           2,
    'skin_splits':        2,
})

DATASET_TO_NAMES = rodict({
    'sun397':             'SUN397',
    'aircraft':           'FGVC-Aircraft',
    'caltech101':         'Caltech101',
    'oxfordpets':         'Oxford Pets-IIIT',
    'flowers102':         'Oxford Flowers',
    'dtd':                'DTD',
    'stanfordcars':       'StanfordCars',
    'voc2007':            'PASCAL VOC2007',
    'brain_tumor_kaggle': 'BrainTumor-Cheng',
    'breakhis':           'BreakHis',
    'skin_splits':        'ISIC 2019',
})

DATASET_TO_LEGEND = rodict({
    'sun397':             'SUN397',
    'aircraft':           'Aircraft',
    'caltech101':         'Cal-101',
    'oxfordpets':         'Ox.Pets',
    'flowers102':         'Ox.Flowers',
    'dtd':                'DTD',
    'stanfordcars':       'Stan.Cars',
    'voc2007':            'VOC 2007',
    'brain_tumor_kaggle': 'BrainTumor',
    'breakhis':           'BreakHis',
    'skin_splits':        'ISIC 2019',
})

MODEL_TO_NAMES = rodict({
    'densenet121':           'DenseNet-121',
    'densenet161':           'DenseNet-161',
    'densenet169':           'DenseNet-169',
    'mobilenetv2_050':       'MobileNet-v2-050',
    'mobilenetv2_100':       'MobileNet-v2-100',
    'resnet18':              'ResNet-18',
    'resnet34':              'ResNet-34',
    'resnet50':              'ResNet-50',
    'efficientnet_b0':       'EfficientNet-B0',
    'vit_small_patch16_224': 'ViT-small-16-224',
})

MODEL_TO_LEGEND = rodict({
    'efficientnet_b0':       'EffNet-B0',
    'densenet121':           'Dense-121',
    'densenet161':           'Dense-161',
    'densenet169':           'Dense-169',
    'mobilenetv2_050':       'Mobil-050',
    'mobilenetv2_100':       'Mobil-100',
    'resnet18':              'ResNet-18',
    'resnet34':              'ResNet-34',
    'resnet50':              'ResNet-50',
    'vit_small_patch16_224': 'ViT-224',
}
)
# This dictionary determines the order of the plots and legend
SINGLE_COLOR = 'dodgerblue'
DATASET_TO_COLOR = rodict({
    # General datasets
    'caltech101': SINGLE_COLOR,
    'sun397':     SINGLE_COLOR,
    'voc2007':    SINGLE_COLOR,

    # Fine-grained datasets: natural
    'flowers102': SINGLE_COLOR,
    'oxfordpets': SINGLE_COLOR,

    # Fine-grained datasets: artificial
    'aircraft':     SINGLE_COLOR,
    'dtd':          SINGLE_COLOR,
    'stanfordcars': SINGLE_COLOR,

    # Medical datasets
    'brain_tumor_kaggle': SINGLE_COLOR,
    'breakhis':           SINGLE_COLOR,
    'skin_splits':        SINGLE_COLOR,
})

DATASET_TO_COLOR = rodict({ d: matplotlib.colors.to_hex(c) for d, c in DATASET_TO_COLOR.items() })


# This dictionary determines the order of the plots and legend
TRANSF_METRICS_TO_MARKERS = rodict({
    'bh3_I+BEST':         'o',
    # 'bh3a_I+BEST':        'D',
    'linear_pooling_overall': 'o',
    'linear_pooling_per_scorer': 'o',
    'linear_pooling_per_score_dataset': 'o',
    'svm': 'o',
    'least_squares': 'o'
})



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
