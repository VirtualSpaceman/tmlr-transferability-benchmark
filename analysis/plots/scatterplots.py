import argparse
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.utils import resample  # pylint: disable=import-error

from ..correlations_compute import fit_regression_model
from ..utils import get_category_colors, normalize_columns, rank_centered


LABEL_REGRETS = False
DEBUG_ON_SUBSET = False

def get_test_logit(score, dataset, multiclass=False): # pylint: disable=unused-argument
    if multiclass:
        # number-of-classes-corrected evidence
        return np.log2(DATASET_CLASSES_N[dataset] * score / (1 - score))
    else:
        # binary 1 vs all evidence
        return np.log2(score / (1 - score))

def plot_regression(ax, x, y, c, m, *, y_raw, title, regret=True, overall=None, x_transform='none',
                    y_transform='none'):
    x = np.asarray(x)
    y = np.asarray(y) * (100 if y_transform == 'none' else 1)
    c = np.asarray(c)
    m = np.asarray(m)

    try:
        # ...generate regression values
        x_reg = np.linspace(min(x), max(x), 100)
        result = fit_regression_model(x, y, margin_x=x_reg, alpha=0.05)

        # ... maximum likelihood prediction intervals for the regression line
        y_reg = result['slope'] * x_reg + result['intercept']
        y_lower = y_reg - result['margin_y']
        y_upper = y_reg + result['margin_y']
        ax.fill_between(x_reg, y_lower, y_upper, color='gray', alpha=0.2)

        # ... maximum likelihood regression line
        ax.plot(x_reg, y_reg, color='red', linewidth=0.5, alpha=0.5)
    except ValueError as e:
        result = x_reg = y_reg = y_lower = y_upper = None
        if 'annot calculate' in str(e):
            ax.text(0.95, 0.05, 'regression failed!', ha='right', va='bottom', color='red',
                    transform=ax.transAxes, fontsize=8)
        else:
            raise e

    # ...data points, grouped by marker
    markers = sorted(set(m))
    for marker in markers:
        subset = m == marker
        # print(x, m, subset, subset.dtype, x.shape, m.shape, subset.shape, sep='\n', end='\n\n')
        x_subset = x[subset]
        y_subset = y[subset]
        c_subset = c[subset]
        ax.scatter(x_subset, y_subset, c=c_subset, marker=marker, alpha=0.5)

    path_stroke = 1.5 if overall == 'scorers' else 1

    # Text label with the regret
    if regret is True:
        regret_label = 'regret'
        regret_value, regret_score = get_regret(x, y_raw)
        regret = f'{regret_label} = {regret_value:.1f} ({regret_score:.2f}σ)'
    else:
        regret_value = regret_score = None
    if regret is not False and LABEL_REGRETS:
        ax.text(0.05, 0.95, regret,
                ha='left', va='top', transform=ax.transAxes, fontsize=8,
                path_effects=[pe.withStroke(linewidth=path_stroke, foreground='white')])

    # Text label with the regression coefficient and its confidence interval
    if result is not None:
        r_maxll = result['rvalue']
        r_mean = np.array(result['bootstrapped_rvalue']).mean()
        r_lower = result['bootstrapped_rvalue_ci'][0]
        r_upper = result['bootstrapped_rvalue_ci'][1]
        if not LABEL_REGRETS:
            ax.text(0.05, 0.95, f'R = {r_mean*100:.0f} [{r_lower*100:.0f} ~ {r_upper*100:.0f}]',
                    ha='left', va='top', transform=ax.transAxes, fontsize=8,
                    path_effects=[pe.withStroke(linewidth=path_stroke, foreground='white')])

        # Once everything is in place, we fix the axes limits and plot the bootstrapped regression lines
        # ... set the axes limits first, because the bootstrapped regression lines may go a bit crazy
        ax.set_xlim(*ax.get_xlim())
        ax.set_ylim(*ax.get_ylim())
        # ... the four extremal bootstrapped regression lines
        for sci in (0, 1):
            for ici in (0, 1):
                y_boots = result['bootstrapped_slope_ci'][sci] * x_reg + result['bootstrapped_intercept_ci'][ici]
                ax.plot(x_reg, y_boots, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    # Axes tick propeties and locators
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    if x_transform == 'none':
        pass
    elif x_transform == 'rank':
        ax.xaxis.set_major_locator(MultipleLocator(1))
    elif x_transform == 'zscore':
        ax.xaxis.set_major_locator(MultipleLocator(1 if overall == 'scorers' else 0.5))
    else:
        assert False, 'Invalid x_transform.'

    if y_transform in ('none', 'logit'):
        pass
    elif y_transform == 'rank':
        ax.yaxis.set_major_locator(MultipleLocator(1))
    elif y_transform == 'zscore':
        ax.yaxis.set_major_locator(MultipleLocator(1 if overall == 'scorers' else 0.5))
    else:
        assert False, 'Invalid y_transform.'

    # Plot title
    ax.text(0.85, 0.05, title, fontstyle='italic', ha='right', va='bottom', transform=ax.transAxes, fontsize=8,
            path_effects=[pe.withStroke(linewidth=path_stroke, foreground='white')])

    # Spines appearance
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return regret_value, regret_score


def get_regret(x, y_raw):
    y_raw = np.asarray(y_raw) * 100
    y_score = (y_raw - np.mean(y_raw)) / np.std(y_raw, ddof=1)
    choice = np.argmax(x)
    best   = np.argmax(y_raw)
    y_choice = y_raw[choice]
    y_best   = y_raw[best]
    y_score_choice = y_score[choice]
    y_score_best   = y_score[best]
    regret_value = y_best - y_choice
    regret_score = y_score_best - y_score_choice
    return regret_value, regret_score


DATASET_CLASSES_N = {
    'sun397'             : 397,
    'aircraft'           : 100,
    'caltech101'         : 102,
    'oxfordpets'         : 37,
    'flowers102'         : 102,
    'dtd'                : 47,
    'stanfordcars'       : 196,
    'voc2007'            : 20,
    'brain_tumor_kaggle' : 4,
    'breakhis'           : 2,
    'skin_splits'        : 2,
}

DATASET_TO_NAMES = {
    'sun397'             : 'SUN397',
    'aircraft'           : 'FGVC-Aircraft',
    'caltech101'         : 'Caltech101',
    'oxfordpets'         : 'Oxford Pets-IIIT',
    'flowers102'         : 'Oxford Flowers',
    'dtd'                : 'DTD',
    'stanfordcars'       : 'StanfordCars',
    'voc2007'            : 'PASCAL VOC2007',
    'brain_tumor_kaggle' : 'BrainTumor-Cheng',
    'breakhis'           : 'BreakHis',
    'skin_splits'        : 'ISIC 2019',
}

DATASET_TO_LEGEND = {
    'sun397'             : 'SUN397',
    'aircraft'           : 'Aircraft',
    'caltech101'         : 'Cal-101',
    'oxfordpets'         : 'Ox.Pets',
    'flowers102'         : 'Ox.Flowers',
    'dtd'                : 'DTD',
    'stanfordcars'       : 'Stan.Cars',
    'voc2007'            : 'VOC 2007',
    'brain_tumor_kaggle' : 'BrainTumor',
    'breakhis'           : 'BreakHis',
    'skin_splits'        : 'ISIC 2019',
}

MODEL_TO_NAMES = {
    'densenet121'          : 'DenseNet-121',
    'densenet161'          : 'DenseNet-161',
    'densenet169'          : 'DenseNet-169',
    'mobilenetv2_050'      : 'MobileNet-v2-050',
    'mobilenetv2_100'      : 'MobileNet-v2-100',
    'resnet18'             : 'ResNet-18',
    'resnet34'             : 'ResNet-34',
    'resnet50'             : 'ResNet-50',
    'efficientnet_b0'      : 'EfficientNet-B0',
    'vit_small_patch16_224': 'ViT-small-16-224',
}

MODEL_TO_LEGEND = {
    'efficientnet_b0'      : 'EffNet-B0',
    'densenet121'          : 'Dense-121',
    'densenet161'          : 'Dense-161',
    'densenet169'          : 'Dense-169',
    'mobilenetv2_050'      : 'Mobil-050',
    'mobilenetv2_100'      : 'Mobil-100',
    'resnet18'             : 'ResNet-18',
    'resnet34'             : 'ResNet-34',
    'resnet50'             : 'ResNet-50',
    'vit_small_patch16_224': 'ViT-224',
}

# This dictionary determines the order of the plots and legend
DATASET_TO_COLOR = {
    # General datasets
    'caltech101'         : 'deeppink',
    'sun397'             : 'hotpink',
    'voc2007'            : 'mediumvioletred',

    # Fine-grained datasets: natural
    'flowers102'         : 'limegreen',
    'oxfordpets'         : 'forestgreen',

    # Fine-grained datasets: artificial
    'aircraft'           : 'skyblue',
    'dtd'                : 'steelblue',
    'stanfordcars'       : 'dodgerblue',


    # Medical datasets
    'brain_tumor_kaggle' : 'orangered',
    'breakhis'           : 'firebrick',
    'skin_splits'        : 'maroon',
}

DATASET_TO_COLOR = { d: matplotlib.colors.to_hex(c) for d, c in DATASET_TO_COLOR.items() }

# This dictionary determines the order of the plots and legend
MODEL_TO_MARKERS = {
    'efficientnet_b0'      : 'o',
    'densenet121'          : 'x',
    'densenet161'          : 'X',
    'densenet169'          : 'P',
    'mobilenetv2_050'      : 's',
    'mobilenetv2_100'      : 'D',
    'resnet18'             : 'v',
    'resnet34'             : '>',
    'resnet50'             : '^',
    'vit_small_patch16_224': '*',
}

# This dictionary determines the order of the plots and legend
TRANSF_METRICS_TO_MARKERS = {
    'bh3_I+BEST':         's',
    'bh3a_I+BEST':        'D',
    'imagenet':           '*',
    'etran_energy_score': 'o',
    'ncti_score':         'd',
    'pactran_score':      'p',
    'gbc_score':          's',
    'parc_score':         '2',
    'leep_score':         'X',
    'nleep_score':        'x',
    'sfda_score':         '^',
    'logme_score':        '>',
    'nce_score':          '<',
    'tmi_score':          'v',
    'hscore_score':       'P',
    'reg_hscore_score':   '+',
}

TRANSF_METRICS_TO_NAMES = {
    'imagenet':           'ImageNet',
    'etran_energy_score': 'ETran',
    'ncti_score':         'NCTI',
    'pactran_score':      'PACTran',
    'gbc_score':          'GBC',
    'parc_score':         'PARC',
    'leep_score':         'LEEP',
    'nleep_score':        'NLEEP',
    'sfda_score':         'SFDA',
    'logme_score':        'LogME',
    'nce_score':          'NCE',
    'tmi_score':          'TMI',
    'hscore_score':       'H-Score',
    'reg_hscore_score':   'R.H-Score',
    'bh3_I+BEST':         'Back to Bayes',
    'bh3a_I+BEST':        'Back to Bayes/Arch.',
}

TRANSF_METRICS_TO_LEGEND = {
    'imagenet':           'ImageNet',
    'etran_energy_score': 'ETran',
    'ncti_score':         'NCTI',
    'pactran_score':      'PACTran',
    'gbc_score':          'GBC',
    'parc_score':         'PARC',
    'leep_score':         'LEEP',
    'nleep_score':        'NLEEP',
    'sfda_score':         'SFDA',
    'logme_score':        'LogME',
    'nce_score':          'NCE',
    'tmi_score':          'TMI',
    'hscore_score':       'H-Score',
    'reg_hscore_score':   'R.H-Score',
    'bh3_I+BEST':         'BtB',
    'bh3a_I+BEST':        'BtB/A',
}

def render_legend(ax, *, marker_items=MODEL_TO_MARKERS, marker_labels=MODEL_TO_LEGEND,
                   color_items=DATASET_TO_COLOR, color_labels=DATASET_TO_LEGEND, ncol=2, axis_info=None, loc='center'):
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
    handles = color_legend + marker_legend
    labels = [h.get_label() for h in handles]
    ax.axis('off')
    ax.legend(handles, labels, loc=loc, fontsize=8, ncol=ncol, frameon=False)
    if axis_info is not None:
        ax.text(0.5, 0.05, axis_info, ha='center', va='bottom', transform=ax.transAxes, fontsize=8)

def main():

    # Create the parser
    parser = argparse.ArgumentParser(description='Scatterplots with classical regressions.')

    # Add the arguments
    parser.add_argument('--input_file', required=True, help='Path to the input .csv file with the experimental data.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory for the plots.')
    parser.add_argument('--x_transform', default='zscore', choices=['rank', 'zscore', 'none'], help='Transform '
                        'to apply to the transfer scores, defaults to zscore.')
    parser.add_argument('--y_transform', default='zscore',
                        choices=['rank', 'zscore', 'logit', 'multiclass_logit', 'none'],
                        help='Transform to apply to the performance metrics, defaults to zscore.')
    parser.add_argument('--normalize_per', choices=['scorer', 'dataset', 'both'], default='both', help='For the zscore '
                        'score, in which groups to perform the z-normalization of the input data. Defaults to both. '
                        'Has no effect if transform is not zscore.')
    parser.add_argument('--skip_metrics', action='store_true', help='If specified, skips the individual metrics plots.')
    parser.add_argument('--skip_overall', action='store_true', help='If specified, skips the aggregated plot with all '
                        'datasets of a scorer.')
    parser.add_argument('--skip_db', action='store_true', help='If specified, skips the aggregated plot with all '
                        'scorers of a dataset.')
    parser.add_argument('--skip_db_R', action='store_true', help='If specified, skips the aggregated plot with all '
                        'scorers of a dataset, with axes reversed.')
    parser.add_argument('--skip_overall_models', action='store_true', help='If specified, skips the aggregated plot '
                        'with scorers and databases of a model.')

    args = parser.parse_args()

    INPUT_PATH = args.input_file
    OUTPUT_DIR = args.output_dir
    X_TRANSFORM = args.x_transform
    Y_TRANSFORM = args.y_transform
    NORMALIZE_PER = args.normalize_per
    PLOT_METRICS = not args.skip_metrics
    PLOT_OVERALL = not args.skip_overall
    PLOT_OVERALL_DB = not args.skip_db
    PLOT_OVERALL_DB_R = not args.skip_db_R
    PLOT_OVERALL_MODELS = not args.skip_overall_models

    if not (PLOT_OVERALL or PLOT_METRICS or PLOT_OVERALL_DB):
        raise ValueError('Nothing to plot.')
    transf_verb = 'Plotting' if PLOT_METRICS else 'Computing'


    # Sample data
    df = pd.read_csv(INPUT_PATH)
    df['symbol'] = df['model'].apply(lambda x: MODEL_TO_MARKERS[x])
    df['color'] = df['dataset'].apply(lambda x: DATASET_TO_COLOR[x])

    if NORMALIZE_PER == 'scorer':
        group_by = ['transf_metric']
    elif NORMALIZE_PER == 'dataset':
        group_by = ['transf_metric']
    elif NORMALIZE_PER == 'both':
        group_by = ['transf_metric', 'dataset']
    else:
        raise ValueError(f'Unknown --normalize_per option: {NORMALIZE_PER}')
    normalize_columns(df, columns=['test_score', 'transf_score'], group_by=group_by)


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
    TRANSF_METRICS_TO_COLORS = {tm : next(colors) for tm in
                                (tm_ for tm_ in transf_metrics if tm_ in TRANSF_METRICS_TO_MARKERS)}

    if PLOT_OVERALL:
        fig2 = plt.figure(figsize=(16, 6.75))
        rows2, cols2 = 4, 4  # Change based on desired layout
        gs = gridspec.GridSpec(rows2, cols2+1, width_ratios=[1] * cols2 + [0.5])
        subplots_n2 = rows2 * cols2
        assert subplots_n2 >= len(TRANSF_METRICS_TO_COLORS), 'Not enough subplots.'
        axes2 = []
        for r in range(rows2):
            for c in range(cols2):
                ax = fig2.add_subplot(gs[r, c])
                axes2.append(ax)
        ax_legend = fig2.add_subplot(gs[:, cols2])
        axes2.append(ax_legend)
        axes2 = iter(axes2)
    else:
        axes2 = iter([])

    db_x = []
    db_y = []
    db_m = []
    db_scorer = []
    db_dataset = []

    db_r = {}
    db_rs = {}

    mod_x = []
    mod_y = []
    mod_c = []
    mod_scorer = []
    mod_model = []


    for transf_metric in TRANSF_METRICS_TO_MARKERS.keys():
        if transf_metric not in transf_metrics:
            continue

        print(f'{transf_verb} {transf_metric}...')

        # Set up the figure and axes
        rows, cols = 3, 4  # Change based on desired layout
        subplots_n = rows * cols
        assert subplots_n >= len(unique_datasets) + 1, 'Not enough subplots.'

        if PLOT_METRICS:
            fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
            fig.subplots_adjust(hspace=0.125, wspace=0.125)
            # fig.subplots_adjust(left=1/32, bottom=1/32, hspace=0.125, wspace=0.125)
            axes = iter(axes.ravel())  # Flatten axes
        else:
            fig = axes = None

        # For each dataset, plot regressions
        all_x = []
        all_y = []
        all_c = []
        all_m = []
        all_r = []
        all_rs = []

        for dataset in DATASET_TO_COLOR.keys():
            if dataset not in unique_datasets:
                continue

            dataset_name = DATASET_TO_NAMES[dataset]
            subset = df[(df['dataset'] == dataset) & (df['transf_metric'] == transf_metric)]
            x = subset['transf_score'].values
            y = subset['test_score'].values
            c = subset['color'].values
            m = subset['symbol'].values
            match X_TRANSFORM:
                case 'none':
                    xt = x
                case 'zscore':
                    xt = subset['z_transf_score'].values
                case 'rank':
                    xt = rank_centered(x)
                case _:
                    raise ValueError('Invalid x_transform.')
            match Y_TRANSFORM:
                case 'none':
                    yt = y
                case 'zscore':
                    yt = subset['z_test_score'].values
                case 'rank':
                    yt = rank_centered(y)
                case 'logit':
                    yt = get_test_logit(y, dataset, multiclass=True)
                case 'multiclass_logit':
                    yt = get_test_logit(y, dataset, multiclass=True)
                case _:
                    raise ValueError('Invalid y_transform.')
            if PLOT_METRICS:
                ax = next(axes)
                regret, regret_score = plot_regression(ax, xt, yt, c, m, y_raw=y, title=dataset_name,
                                                       x_transform=X_TRANSFORM, y_transform=Y_TRANSFORM)
            else:
                regret, regret_score = get_regret(xt, y)

            n = len(xt)
            assert n == len(yt) == len(c) == len(m) == len(subset)

            all_x.extend(xt)
            all_y.extend(yt)
            all_c.extend(c)
            all_m.extend(m)

            db_x.extend(xt)
            db_y.extend(yt)
            db_m.extend(m)
            db_scorer.extend([transf_metric] * n)
            db_dataset.extend([dataset] * n)

            mod_x.extend(xt)
            mod_y.extend(yt)
            mod_c.extend(c)
            mod_scorer.extend([transf_metric] * n)
            mod_model.extend(subset['model'].values)

            db_r.setdefault(dataset,[]).append(regret)
            db_rs.setdefault(dataset,[]).append(regret_score)
            all_r.append(regret)
            all_rs.append(regret_score)

        # print(all_m)
        # Plot overall regression
        reg_avg = np.mean(all_r)
        reg_score_avg = np.mean(all_rs)
        reg_max = np.max(all_r)
        reg_score_max = np.max(all_rs)
        regret = f'r.avg.={reg_avg:.1f}|{reg_score_avg:.2f}σ ~ r.max.={reg_max:.1f}|{reg_score_max:.2f}σ'
        if PLOT_METRICS:
            # Plots aggregate regression
            ax = next(axes)
            plot_regression(ax, all_x, all_y, all_c, all_m, y_raw=[], title='Combined', regret=regret,
                            x_transform=X_TRANSFORM, y_transform=Y_TRANSFORM)

            # Turn off any unused subplots
            for ax in axes:
                ax.axis('off')

            # Titles the figure
            # if TRANSFORM == 'rank':
            #     # TODO: the overall plot will be mislabelled - it has the z-scores
            #     xr, yr = rank_centered(x), rank_centered(y)
            #     x_axis_label = f'Rank({transf_metric})'
            #     y_axis_label = 'Rank(Test Metric)'
            # elif TRANSFORM == 'zscore':
            #     x_axis_label = f'z-score({transf_metric})'
            #     y_axis_label = 'z-score(Test Metric)'
            # elif TRANSFORM == 'none':
            #     # TODO: the overall plot will be mislabelled - it has the z-scores
            #     x_axis_label = f'{transf_metric}'
            #     y_axis_label = 'Test Metric'
            # else:
            #     raise ValueError('Invalid transform.')
            # title = f'{x_axis_label} vs {y_axis_label}'
            # fig.text(0.5, 1/64, x_axis_label, ha='center', va='center')
            # fig.text(1/64, 0.5, y_axis_label, ha='center', va='center', rotation='vertical')
            # fig.suptitle(title, fontsize=10)

            fig.savefig(os.path.join(OUTPUT_DIR, f'plot_{transf_metric}.pdf'), bbox_inches='tight')
            plt.close(fig)

        if PLOT_OVERALL:
            # Plots aggregate regression
            ax2 = next(axes2)
            title = TRANSF_METRICS_TO_NAMES[transf_metric]
            plot_regression(ax2, all_x, all_y, all_c, all_m, y_raw=[], title=title, regret=regret,
                            overall='scorers', x_transform=X_TRANSFORM, y_transform=Y_TRANSFORM)

    if PLOT_OVERALL:
        print('Saving overall plot...')
        ax2 = next(axes2)
        render_legend(ax2, ncol=1, axis_info='x = Transfer Scores\ny = Test Metrics\n\n'
                      'both axes z-normalized\nper scorer & dataset', loc='upper center')
        for ax2 in axes2:
            ax2.axis('off')
        fig2.savefig(os.path.join(OUTPUT_DIR, 'plot_scorers.pdf'), bbox_inches='tight')
        plt.close(fig2)

    db_x = np.asarray(db_x)
    db_y = np.asarray(db_y)
    db_m = np.asarray(db_m)
    db_scorer = np.asarray(db_scorer)
    db_dataset = np.asarray(db_dataset)

    dataset_plots = []
    if PLOT_OVERALL_DB:
        dataset_plots.append('datasets')
        dataset_plots.append('datasets_clean')
    if PLOT_OVERALL_DB_R:
        dataset_plots.append('datasets_clean_reversed')
        dataset_plots.append('datasets_reversed')

    for dataset_plot in dataset_plots:
        print(f'Saving {dataset_plot} plot...')
        rows, cols = 3, 4  # Change based on desired layout
        subplots_n = rows * cols
        assert subplots_n >= len(unique_datasets), 'Not enough subplots.'
        fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
        fig.subplots_adjust(hspace=0.125, wspace=0.125)
        axes = iter(axes.ravel())  # Flatten axes
        for dataset in DATASET_TO_COLOR.keys():
            if dataset not in unique_datasets:
                continue
            color_items = dict(TRANSF_METRICS_TO_COLORS)
            if '_clean' in dataset_plot:
                this_dataset = (db_dataset == dataset) & (~np.isin(db_scorer, ('tmi_score', 'hscore_score', 'reg_hscore_score')))
                color_items.pop('tmi_score', None)
                color_items.pop('hscore_score', None)
                color_items.pop('reg_hscore_score', None)
            else:
                this_dataset = db_dataset == dataset
            ax = next(axes)
            dataset_name = DATASET_TO_NAMES[dataset]

            x = db_x[this_dataset]
            y = db_y[this_dataset]
            c = [TRANSF_METRICS_TO_COLORS[tm] for tm in db_scorer[this_dataset]]
            m = db_m[this_dataset]
            xt, yt = (y, x) if 'reversed' in dataset_plot else (x, y)

            reg_avg = np.mean(db_r[dataset])
            reg_score_avg = np.mean(db_rs[dataset])
            reg_max = np.max(db_r[dataset])
            reg_score_max = np.max(db_rs[dataset])
            regret = f'r.avg.={reg_avg:.1f}|{reg_score_avg:.2f}σ ~ r.max.={reg_max:.1f}|{reg_score_max:.2f}σ'

            plot_regression(ax, xt, yt, c, m, y_raw=[], title=dataset_name, regret=regret, x_transform=X_TRANSFORM,
                            y_transform=Y_TRANSFORM, overall='datasets')
        ax = next(axes)
        render_legend(ax, color_items=color_items, color_labels=TRANSF_METRICS_TO_LEGEND)
        for ax in axes:
            ax.axis('off')
        fig.savefig(os.path.join(OUTPUT_DIR, f'plot_{dataset_plot}.pdf'), bbox_inches='tight')
        plt.close(fig)


    if PLOT_OVERALL_MODELS:
        mod_x = np.asarray(mod_x)
        mod_y = np.asarray(mod_y)
        mod_c = np.asarray(mod_c)
        mod_scorer = np.asarray(mod_scorer)
        mod_model = np.asarray(mod_model)

        for model_plot in ('models', 'models_clean'):
            print(f'Saving {model_plot} plot...')
            rows, cols = 3, 4  # Change based on desired layout
            subplots_n = rows * cols
            assert subplots_n >= len(unique_models), 'Not enough subplots.'
            fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
            fig.subplots_adjust(hspace=0.125, wspace=0.125)
            axes = iter(axes.ravel())  # Flatten axes
            for model in MODEL_TO_MARKERS.keys():
                if model not in unique_models:
                    continue

                marker_items = dict(TRANSF_METRICS_TO_MARKERS)
                if '_clean' in model_plot:
                    this_model = (mod_model == model) & (~np.isin(mod_scorer, ('tmi_score', 'hscore_score', 'reg_hscore_score')))
                    marker_items.pop('tmi_score', None)
                    marker_items.pop('hscore_score', None)
                    marker_items.pop('reg_hscore_score', None)
                else:
                    this_model = mod_model == model
                ax = next(axes)
                model_name = MODEL_TO_NAMES[model]
                x = mod_x[this_model]
                y = mod_y[this_model]
                c = mod_c[this_model] # datasets
                m = [TRANSF_METRICS_TO_MARKERS[tm] for tm in mod_scorer[this_model]]
                plot_regression(ax, y, x, c, m, y_raw=[], title=model_name, regret='', x_transform=X_TRANSFORM,
                                y_transform=Y_TRANSFORM, overall='models')
            ax = next(axes)
            render_legend(ax, marker_items=marker_items, marker_labels=TRANSF_METRICS_TO_LEGEND)
            for ax in axes:
                ax.axis('off')
            fig.savefig(os.path.join(OUTPUT_DIR, f'plot_{model_plot}_reversed.pdf'), bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    main()
