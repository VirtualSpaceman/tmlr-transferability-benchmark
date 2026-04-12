
import argparse
import json
from types import MappingProxyType as rodict

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from ..utils import get_category_colors, rank_jittered


LABEL_REGRETS = False
DEBUG_ON_SUBSET = False


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
DATASET_TO_COLOR = rodict({
    # General datasets
    'caltech101': 'deeppink',
    'sun397':     'hotpink',
    'voc2007':    'mediumvioletred',

    # Fine-grained datasets: natural
    'flowers102': 'limegreen',
    'oxfordpets': 'forestgreen',

    # Fine-grained datasets: artificial
    'aircraft':     'skyblue',
    'dtd':          'steelblue',
    'stanfordcars': 'dodgerblue',


    # Medical datasets
    'brain_tumor_kaggle': 'orangered',
    'breakhis':           'firebrick',
    'skin_splits':        'maroon',
})

DATASET_TO_COLOR = rodict({ d: matplotlib.colors.to_hex(c) for d, c in DATASET_TO_COLOR.items() })

# This dictionary determines the order of the plots and legend
MODEL_TO_MARKERS = rodict({
    'efficientnet_b0':       'o',
    'densenet121':           'x',
    'densenet161':           'X',
    'densenet169':           'P',
    'mobilenetv2_050':       's',
    'mobilenetv2_100':       'D',
    'resnet18':              'v',
    'resnet34':              '>',
    'resnet50':              '^',
    'vit_small_patch16_224': '*',
})

# This dictionary determines the order of the plots and legend
TRANSF_METRICS_TO_MARKERS = rodict({
    # 'bh3_I+BEST':         's',
    # 'bh3a_I+BEST':        'D',
    'per_scorer_dataset_scorer': 's',
    'imagenet':           '*',
    'ncti_score':         'd',
    'etran_energy_score': 'o',
    'pactran_score':      'p',
    'gbc_score':          's',
    'parc_score':         '2',
    'nleep_score':        'x',
    'leep_score':         'X',
    'logme_score':        '>',
    'tmi_score':          'v',
    'nce_score':          '<',
    'sfda_score':         '^',
    'hscore_score':       'P',
    'reg_hscore_score':   '+',
})

TRANSF_METRICS_TO_NAMES = rodict({
    'bh3_I+BEST':         'Back to Bayes',
    # 'bh3a_I+BEST':        'Back to Bayes/Arch.',
    'imagenet':           'ImageNet',
    'ncti_score':         'NCTI',
    'etran_energy_score': 'ETran',
    'pactran_score':      'PACTran',
    'gbc_score':          'GBC',
    'parc_score':         'PARC',
    'nleep_score':        'NLEEP',
    'leep_score':         'LEEP',
    'logme_score':        'LogME',
    'tmi_score':          'TMI',
    'nce_score':          'NCE',
    'sfda_score':         'SFDA',
    'hscore_score':       'H-Score',
    'reg_hscore_score':   'R.H-Score',
})

TRANSF_METRICS_TO_LEGEND = rodict({
    'bh3_I+BEST':         'BtB',
    # 'bh3a_I+BEST':        'BtB/A',
    'imagenet':           'ImageNet',
    'ncti_score':         'NCTI',
    'etran_energy_score': 'ETran',
    'pactran_score':      'PACTran',
    'gbc_score':          'GBC',
    'parc_score':         'PARC',
    'nleep_score':        'NLEEP',
    'leep_score':         'LEEP',
    'logme_score':        'LogME',
    'tmi_score':          'TMI',
    'nce_score':          'NCE',
    'sfda_score':         'SFDA',
    'hscore_score':       'H-Score',
    'reg_hscore_score':   'R.H-Score',
})


def create_subplot(ax, x, y, c, m, *, stat=None, title=None):
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)
    m = np.asarray(m)

    # ...data points, grouped by marker
    markers = sorted(set(m))
    for marker in markers:
        subset = m == marker
        x_subset = x[subset]
        y_subset = y[subset]
        c_subset = c[subset]
        ax.scatter(x_subset, y_subset, c=c_subset, marker=marker, alpha=0.5)

    path_stroke = 2.5

    # Axes tick propeties and locators
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Annotated statistic
    if stat:
        ax.text(0.05, 0.95, f'{stat["symbol"]} = {stat["mean"]*100:.0f} '
                f'[{stat["ci"][0]*100:.0f} ~ {stat["ci"][1]*100:.0f}]',
                ha='left', va='top', transform=ax.transAxes, fontsize=8,
                path_effects=[pe.withStroke(linewidth=path_stroke, foreground='white')])

    # Plot title
    if title:
        ax.text(0.95, 0.05, title, fontstyle='italic', ha='right', va='bottom', transform=ax.transAxes, fontsize=8,
                path_effects=[pe.withStroke(linewidth=path_stroke, foreground='white')])

    # Spines appearance
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def render_legend(axes, *, marker_items=MODEL_TO_MARKERS, marker_labels=MODEL_TO_LEGEND, marker_title=r'$\it{Models}$',
                  color_items=DATASET_TO_COLOR, color_labels=DATASET_TO_LEGEND, color_title=r'$\it{Datasets}$',
                  ncol=2, axis_info=None, loc='center'):

    ax = next(axes)
    ax.axis('off')
    if axis_info is not None:
        ax.text(0.5, 0.5, axis_info, ha='center', va='center', transform=ax.transAxes, fontsize=9)

    # Creating patches for the color legend
    color_legend = []
    for color_item, color in color_items.items():
        color_label = color_item if color_labels is None else color_labels[color_item]
        color_legend.append(mpatches.Patch(color=color, label=color_label))

    # Creating patches for the marker legend
    marker_legend = []
    for marker_item, marker in marker_items.items():
        marker_legend.append(plt.Line2D([0], [0], marker=marker, color='black', markerfacecolor='gray', markersize=7,
                             linestyle='None', label=marker_labels[marker_item]))

    # Displaying the legends on the axes
    color_patch = mpatches.Patch(color='none', label=color_title)
    spacing_patch = mpatches.Patch(color='none', label='')
    marker_patch = mpatches.Patch(color='none', label=marker_title)
    handles = [color_patch] + color_legend + [spacing_patch, marker_patch] + marker_legend
    labels = [h.get_label() for h in handles]

    ax = next(axes)
    ax.axis('off')
    ax.legend(handles, labels, loc=loc, fontsize=8, ncol=ncol, frameon=False)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Scatterplots with classical regressions.')

    # Add the arguments
    parser.add_argument('--input_file', required=True, help='Path to the input .csv file with the experimental data.')
    parser.add_argument('--input_boots', required=True, help="Path to the input .json file with the bootstrapping of "
                        "the grouped weighted Kendall\'s tau statistic.")
    parser.add_argument('--output_file', required=True, help='Path to the output plot file (format inferred from '
                        'file extension, e.g., .pdf, .png, .svg).')
    args = parser.parse_args()

    INPUT_PATH = args.input_file
    BOOTS_PATH = args.input_boots
    OUTPUT_FILE = args.output_file

    # Sample data
    df = pd.read_csv(INPUT_PATH)
    df['symbol'] = df['model'].apply(lambda x: MODEL_TO_MARKERS[x])
    df['color'] = df['dataset'].apply(lambda x: DATASET_TO_COLOR[x])

    # Bootstrapping statistics
    with open(BOOTS_PATH, 'rt', encoding='utf-8') as f:
        wtau = json.load(f)

    # Test on a subset of the data
    if DEBUG_ON_SUBSET:
        df = df[(df['dataset'].isin(['sun397', 'aircraft'])) & (df['transf_metric'].isin(['imagenet', 'nce_score']))]

    unique_datasets = df['dataset'].unique()
    if not set(unique_datasets).issubset(set(DATASET_CLASSES_N.keys())):
        raise ValueError('Dataset names do not match.')

    transf_metrics = sorted(df['transf_metric'].unique())
    unique_models = df['model'].unique()
    if not set(unique_models).issubset(set(MODEL_TO_NAMES.keys())):
        raise ValueError('Model names do not match.')

    colors = get_category_colors()
    TRANSF_METRICS_TO_COLORS = {tm: next(colors) for tm in
                                (tm_ for tm_ in transf_metrics if tm_ in TRANSF_METRICS_TO_MARKERS)}

    mm_to_in = (1./25.4)*4
    fig = plt.figure(figsize=(82*mm_to_in, 48*mm_to_in))
    rows, cols = 4, 4  # Change based on desired layout
    gs = gridspec.GridSpec(rows, cols+1, width_ratios=[1] * cols + [0.5])
    subplots_n2 = rows * cols
    assert subplots_n2 >= len(TRANSF_METRICS_TO_COLORS), 'Not enough subplots.'
    axes = []
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(gs[r, c])
            axes.append(ax)
    ax_legend = fig.add_subplot(gs[:, cols])
    axes.append(ax_legend)
    axes = iter(axes)

    for transf_metric in TRANSF_METRICS_TO_MARKERS:
        if transf_metric not in transf_metrics:
            continue

        print(f'{transf_metric}...')
        # For each dataset, plot regressions
        all_x = []
        all_y = []
        all_c = []
        all_m = []
        all_d = []

        for d, dataset in enumerate(DATASET_TO_COLOR):
            if dataset not in unique_datasets:
                continue
            subset = df[(df['dataset'] == dataset) & (df['transf_metric'] == transf_metric)]
            x = subset['transf_score'].values
            y = subset['test_score'].values
            c = subset['color'].values
            m = subset['symbol'].values
            assert len(x) == len(y) == len(c) == len(m) == len(subset)

            xt = rank_jittered(x)
            yt = rank_jittered(y)

            all_x.extend(xt)
            all_y.extend(yt)
            all_c.extend(c)
            all_m.extend(m)
            all_d.extend([d] * len(x))

        # Plots aggregate regression
        ax = next(axes)
        title = TRANSF_METRICS_TO_NAMES[transf_metric]
        # TODO: this is indexed as 'tau' but is the 'wtau'
        tau_maxll = wtau[transf_metric]['tau']
        tau_ci = wtau[transf_metric]['ci']
        tau_mean = np.mean(wtau[transf_metric]['boots'])
        tau = dict(symbol=r'$\mathring{\tau_\text{w}}$', maxll=tau_maxll, ci=tau_ci, mean=tau_mean)
        create_subplot(ax, all_x, all_y, all_c, all_m, stat=tau, title=title)

    print('Saving overall plot...')
    info  = 'x-axis = rank|(|Transfer Scores|)\ny-axis = rank|(|Test Metrics|)'.replace('|', '\u2009')  # Thin spaces
    info += '\n(ranks computed per-dataset)'
    info += '\n\n' + r"$\mathring{\tau_\text{w}}$ = aggregated weighted tau $\times$ 100"
    info += '\nmean [95% CI] of 1000 bootstraps'
    render_legend(axes, ncol=1, axis_info=info, loc='upper center')
    for ax in axes:
        ax.axis('off')
    fig.savefig(OUTPUT_FILE, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
