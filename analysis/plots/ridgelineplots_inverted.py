import argparse
import colorsys
import glob
from itertools import chain
import json
import os
import sys
import copy
import arviz as az
import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from scipy.stats import gaussian_kde


HDI_TAIL_PROB = 0.05
SCRIPT_FILE = os.path.abspath(__file__)
SCRIPT_NAME = os.path.splitext(os.path.basename(SCRIPT_FILE))[0]
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)
LAYOUT_FILE = os.path.join(SCRIPT_DIR, f'{SCRIPT_NAME}.json')

DPI = 200

PREFIX_JITTER = 0  # 0.05
PREFIX_FONT_JITTER = 0.0  # 0.1
# MAIN_FONT = 'TeX Gyre Termes'
# PREFIX_FONTS = ['TeX Gyre Termes', 'Liberation Serif', 'STIX Two Text']

# Create the parser
parser = argparse.ArgumentParser(description='Classical regression model for a single transfer metric.')

# Add the arguments
# parser.add_argument('--input_file', required=True, help='Path to the input .csv file with the experimental data.')
parser.add_argument('--input_dirs', required=True, nargs='+', help='Paths to the input directories with the .json '
                    'summaries created by summarize.nf, or "!" to simulate random data.')
parser.add_argument('--input_layout', default=LAYOUT_FILE, help='Path to input .json file with the layout of the plot '
                    'to generate. If omitted, loads one of the default layouts.')
parser.add_argument('--output_file', required=True, help='Path to the output plot. The file extension (.pdf, .png, '
                    'etc.) determines the image format. Use "!" to show the plot instead of saving it.')
parser.add_argument('--summary', default='correlations', help='Which summary to exploit. Defaults to "correlations".')
parser.add_argument('--measurement', default='wtau', help='Which measurement to plot. Typical values are "tau", "wtau". '
                    'Defaults to "wtau".')
parser.add_argument('--statistic', default='mean', help='Which statistic to plot. Typical values are "mean" for the '
                    'mean of the samples, or "maxll" for the pure measurement from data. Not all statistics are '
                    'available for all combinations of summary and measurement. Defaults to "mean".')
parser.add_argument('--plot_type', default='kde', choices=['kde', 'hist', 'samples'], help='Plot type: choose among '
                    '"kde" for a kernel density estimate, "hist" for a histogram, or "samples" for the jittered '
                    'raw samples. Defaults to "kde".')
parser.add_argument('--steady_samples', action='store_true', help='If set, the samples will be plotted exactly as '
                    'they were generated, without any jittering. This is only available for --plot_type samples.')
parser.add_argument('--hdi_tail_prob', default=HDI_TAIL_PROB, type=float, help='Tail probability for the HDI used to '
                    f'compute the x-axis limits of kde plots. Defaults to {HDI_TAIL_PROB}.')
parser.add_argument('--number_format', default='.0f', help='Format string for the numbers in the plot. Defaults to '
                    '".0f".')
parser.add_argument('--number_scaling', default=100, type=float, help='Scaling factor for the numbers in the plot. '
                    'Defaults to 100.')
parser.add_argument('--y_bleed', default=0.2, type=float, help='How much the "regular" y-axis should bleed into the '
                    'previous row, in a fraction of the row height. Defaults to 0.1.')
parser.add_argument('--y_crop', default=0.8, type=float, help='How much the entire y-axis should be allowed to bleed '
                    'into the previous rows, as a fraction of the row height. Defaults to 0.9.')
parser.add_argument('--dpi', default=DPI, help=f'DPI of the output image. Default: {DPI}.')
parser.add_argument('--vectorized_scatter', action='store_true', help='If set, the scatter plots will not be '
                    'pre-rasterized. This may result in large-files and slow rendering, but will allow the scatter '
                    'plots to be zoomed in without pixelation.')

args = parser.parse_args()

# INPUT_FILE = args.input_file
INPUT_DIRS = args.input_dirs
INPUT_LAYOUT = args.input_layout
OUTPUT_FILE = args.output_file
SUMMARY = args.summary
MEASUREMENT = args.measurement
STATISTIC = args.statistic
PLOT_TYPE = args.plot_type
STEADY_SAMPLES = args.steady_samples
HDI_TAIL_PROB = args.hdi_tail_prob
NUMBER_FORMAT = args.number_format
NUMBER_SCALING = args.number_scaling
Y_BLEED = args.y_bleed
Y_CROP = args.y_crop
DPI = args.dpi
VECTORIZED_SCATTER = args.vectorized_scatter

if Y_BLEED < 0:
    raise ValueError(f'--y_bleed must be non-negative, but {Y_BLEED} was given.')
if Y_CROP < Y_BLEED:
    raise ValueError(f'--y_crop must be greater than or equal to -y_bleed, but {Y_CROP} was given.')

# Reads input data
# df = pd.read_csv(INPUT_FILE)
# datasets = df['dataset'].unique()
# scorers = df['transf_metric'].unique()

# Reads layout file
with open(INPUT_LAYOUT, 'rt', encoding='utf-8') as f:
    layout = json.load(f)

summarize_all = layout['prefix_summarize_all']

# Columns subset
cols_subset = [x for x in layout['cols'] if x['header'] in ['Combined', 'Averaged']]
rows_subset = copy.deepcopy(layout['rows'])

# Change rows with columns 
n_rows = len(layout['rows'])
n_columns = len(layout['cols'])
all_subsamples = {}
all_stats = {}
all_hdis = {}
all_kdes = {}
all_histograms = {}
y_max = []

# Set random seed for reproducibility
np.random.seed(0)

# consolidates input directories
n_prefixes = len(layout['rows'][0]['prefixes'])
averaged = [[[] for _ in range(n_prefixes)] for _ in range(n_rows)]
summaries = {}
for input_dir in INPUT_DIRS:
    for input_path in glob.glob(os.path.join(input_dir, 'summ_*.json')):
        input_file = os.path.basename(input_path)
        if input_file in summaries:
            print(f'WARNING: duplicate summary file {input_file} found both in {summaries[input_file]} and '
                    f'{input_path}: only the latter will be considered.', file=sys.stderr)
        summaries[input_file] = input_path
# creates a distribution for each cell
print(f"{summaries=}")
x_grids = []
for c, col in enumerate(layout['cols']):
    x_min = np.inf
    x_max = -np.inf
    for r, row in enumerate(layout['rows']):
        scorer = row['scorer']
        if len(row['prefixes']) != n_prefixes:
            raise ValueError(f'Row: {r} -- all rows must have the same number of prefixes: expected {n_prefixes}, '
                                f'found {len(row["prefixes"])}.')
        for p, prefix in enumerate(row['prefixes']):
            if prefix is None:
                all_subsamples.setdefault((r, c), []).append(None)
                all_stats.setdefault((r, c), []).append(None)
                all_hdis.setdefault((r, c), []).append(None)
                all_kdes.setdefault((r, c), []).append(None)
                all_histograms.setdefault((r, c), []).append(None)
            else:
                if 'dataset' in col:
                    dataset = col['dataset']
                    scorer_file = f'summ_{SUMMARY}_{prefix}{scorer}.json'
                    if scorer_file not in summaries:
                        raise ValueError(f'Could not find summary file {scorer_file} in any input directory.')
                    with open(summaries[scorer_file], 'rt', encoding='utf-8') as f:
                        summary = json.load(f)
                    hdi = summary[dataset][MEASUREMENT]['HDI']
                    samples = np.asarray(summary[dataset][MEASUREMENT]['samples'])
                    # Check for na and inf
                    if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
                        print(f'WARNING: NaN or Inf found in samples of summ_{SUMMARY}_{prefix}{scorer}/{dataset}',
                                file=sys.stderr)
                    averaged[r][p].append(samples)
                    all_stats.setdefault((r, c), []).append(summary[dataset][MEASUREMENT])
                elif 'summary' in col:
                    col_name = col['summary']
                    samples = np.asarray(list(chain.from_iterable(averaged[r][p])))
                    # This is the mean of the samples, not of the statistics, so the code currently doesn't work
                    # for the other statistics.
                    mean = samples.mean()
                    all_stats.setdefault((r, c), []).append({'mean': mean, 'maxll': None, 'mean-maxll': None})
                else:
                    raise ValueError(f'Column: {c} -- each column must have either a "dataset" or a "summary" '
                                        f'attribute.')
                if PLOT_TYPE == 'hist':
                    subsamples = kde = None
                    hist, bins = np.histogram(samples, bins='auto', density=True)
                    x_min = min(x_min, *bins)
                    x_max = max(x_max, *bins)
                    ym = np.max(hist)
                elif PLOT_TYPE == 'kde':
                    subsamples = hist = bins = None
                    # print(samples)
                    kde = gaussian_kde(samples)
                    x_hdi = az.hdi(samples, hdi_prob=1.-HDI_TAIL_PROB)
                    x_min = min(x_min, x_hdi[0])
                    x_max = max(x_max, x_hdi[1])
                    x_grid = np.linspace(x_hdi[0], x_hdi[1], 100)
                    kde_grid = kde.pdf(x_grid)
                    ym = np.max(kde_grid)
                elif PLOT_TYPE == 'samples':
                    hist = bins = kde = None
                    subsamples = np.random.choice(samples, size=100, replace=False) if len(samples) > 100 \
                                                                                    else samples
                    x_min = min(x_min, *samples)
                    x_max = max(x_max, *samples)
                    ym = 1
                else:
                    assert False
                y_max.append(ym)
                all_subsamples.setdefault((r, c), []).append(subsamples)
                all_hdis.setdefault((r, c), []).append(hdi)
                all_kdes.setdefault((r, c), []).append(kde)
                all_histograms.setdefault((r, c), []).append((hist, bins))
    print(c, x_min, x_max)
    x_grids.append(np.linspace(x_min, x_max, 100))

print(f"{all_stats.keys()}=") # 182 keys ->  14 * 13  -> scorer x datasets (até id 10) + average (11) + combined (12)
# reads input distributions
y_max = np.asarray(y_max)
y_max_max = np.max(y_max)
y_max_q90 = np.quantile(y_max, 0.9)
y_max_q75 = np.quantile(y_max, 0.75)
y_max_q50 = np.quantile(y_max, 0.5)
y_max_choice = y_max_q75
print(f'Max. KDE quantiles: {y_max_max:.2f} (max), {y_max_q90:.2f} (90%), {y_max_q75:.2f} (75%), {y_max_q50:.2f} (50%)')

# Plot appearance: the font metrics will scale up or down the entire plot
fontsize = 35
fontsize_data = 30 if summarize_all else 31
row_factor = 1.5
col_factor = 1

# ...all those parameters in figsize-like inches:
figure_h_padding = 0.25
figure_w_padding = 0.25
row_spacing = 0.5
col_spacing = 0.05

# plt.rcParams['font.size'] = fontsize
# plt.rcParams["text.usetex"] = True
# plt.rcParams['font.family'] = MAIN_FONT

# Change rows with columns 
aux = rows_subset
rows_subset = cols_subset # averaged / combined 
cols_subset = aux # Scorers 
n_columns = len(cols_subset)
n_rows = len(rows_subset)

print(f"{rows_subset=}")
print(f"{cols_subset=}")


# Get font metrics to dimension the figure
fig = plt.figure(figsize=(10, 1), dpi=DPI, layout='none')
col_width = row_height = 0
# ...finds widest and tallest column header
for c in cols_subset:
    text = fig.text(0.5, 0.5, c["header"], fontsize=fontsize, verticalalignment='center', horizontalalignment='center')
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer)
    bbox = fig.dpi_scale_trans.inverted().transform_bbox(bbox)  # pixels -> figsize coordinates
    col_width = max(col_width, bbox.width)
    row_height = max(row_height, bbox.height)
    text.remove()

col_header_width = col_width
# ...finds widest and tallest row header
row_header_width = 0
for r in rows_subset:
    text = fig.text(0.5, 0.5, r["header"], fontsize=fontsize, verticalalignment='center', horizontalalignment='center')
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer)
    bbox = fig.dpi_scale_trans.inverted().transform_bbox(bbox)  # pixels -> figsize coordinates
    row_header_width = max(row_header_width, bbox.width)
    row_height = max(row_height, bbox.height)
    text.remove()
plt.close()
print(f'Col Width: {col_width}, Row Height: {row_height}, Row Header Width: {row_header_width}')

# Initialize the figure and subplots again
table_width = n_columns * col_width * col_factor + (n_columns - 1) * col_spacing + figure_w_padding * 2
table_height = n_rows * row_height * row_factor + (n_rows - 1) * row_spacing + figure_h_padding * 2
print(f'Table Width: {table_width}, Table Height: {table_height}')

fig = plt.figure(figsize=(table_width, table_height), dpi=DPI, layout='none')


def to_figure_width(w):
    display_bbox = fig.dpi_scale_trans.transform_bbox(Bbox.from_extents((0, 0, w, 0)))
    figure_bbox = fig.transFigure.inverted().transform_bbox(display_bbox)
    return figure_bbox.width


def to_figure_height(h):
    display_bbox = fig.dpi_scale_trans.transform_bbox(Bbox.from_extents((0, 0, 0, h)))
    figure_bbox = fig.transFigure.inverted().transform_bbox(display_bbox)
    return figure_bbox.height


def to_axis_width(ax, w):
    display_bbox = fig.dpi_scale_trans.transform_bbox(Bbox.from_extents((0, 0, w, 0)))
    axis_bbox = ax.transAxes.inverted().transform_bbox(display_bbox)
    return axis_bbox.width


def to_data_width(ax, w):
    display_bbox = fig.dpi_scale_trans.transform_bbox(Bbox.from_extents((0, 0, w, 0)))
    data_bbox = ax.transData.inverted().transform_bbox(display_bbox)
    return data_bbox.width


col_spacing_inches = col_spacing
row_header_width = to_figure_width(row_header_width)
col_width = to_figure_width(col_width)
row_height = to_figure_height(row_height)
figure_w_padding = to_figure_width(figure_w_padding)
figure_h_padding = to_figure_height(figure_h_padding)
col_spacing = to_figure_width(col_spacing)
row_spacing = to_figure_height(row_spacing)

print('row_header_width', row_header_width)
print('col_width', col_width)
print('row_height', row_height)
print('figure_w_padding', figure_w_padding)
print('figure_h_padding', figure_h_padding)
print('col_spacing', col_spacing)
print('row_spacing', row_spacing)

# Computes spaces in figure coordinates (0 to 1, from let to right, bottom to top)
# ... specifications
table_cols = n_columns + 1
table_rows = n_rows + 1

# ... computations
col_widths = np.asarray([row_header_width * 0.77] + [col_width] * n_columns)  # TODO: fix this ugly manual adjustment
col_lefts = np.cumsum([figure_w_padding] + list(col_widths[:-1] + col_spacing))
row_heights = np.asarray([row_height] * table_rows)  # Rows from bottom to top
row_bottoms = np.cumsum([figure_h_padding] + list(row_heights[:-1] + row_spacing))  # Rows from bottom to top

# ... cell heights considering overlaps (bleeds) and crops
if PLOT_TYPE == 'samples':
    Y_BLEED = Y_CROP = 0
full_height = row_height + row_spacing
crop_height = full_height * (1. + Y_CROP)
bleed_height = full_height * (1. + Y_BLEED)
y_max_bleed = y_max_choice  # y limit up to the bleeding height
y_max_crop = y_max_bleed / bleed_height * crop_height  # y limit up to the cropping height
y_max_full = y_max_bleed / bleed_height * full_height  # y limit up to the original row height (with spacing)
y_max_row = y_max_bleed / bleed_height * row_height  # y limit up to the original row height (w/o spacing)
h_max_row = row_height / crop_height

print('row_height', row_height)
print('full_height', full_height)
print('crop_height', crop_height)
print('bleed_height', bleed_height)
print('y_max_choice', y_max_choice)
print('y_max_bleed', y_max_bleed)
print('y_max_crop', y_max_crop)
print('y_max_row', y_max_row)

# ... facility function
def get_ax(row, col):  # Rows from top to bottom
    header = row == 0 or col == 0
    row = table_rows - row - 1
    if header:
        # Column headers and row headers have exactly the row height
        h = row_heights[row]
        y_max = None
    else:
        # Data cells
        h = crop_height
        y_max = y_max_crop

    position = [col_lefts[col], row_bottoms[row], col_widths[col], h]
    ax = fig.add_axes(position, facecolor='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if y_max is not None:
        ax.set_ylim(0, y_max)

    return ax


# col_lefts = np.cumsum([figure_w_padding] + list(col_widths[:-1] + col_spacing))
# row_bottoms = np.cumsum([figure_h_padding] + list(row_heights[:-1] + row_spacing)) # Rows from bottom to top
mid_lines = 0.75
outer_lines = 1.5


def add_lines():
    x_eps = col_spacing / 2
    x_left = col_lefts[0]
    x_right = col_lefts[-1] + col_widths[-1]

    y_eps = row_spacing / 2
    y_bottom = row_bottoms[0] - y_eps
    y_top = row_bottoms[-1] + row_heights[-1] + row_spacing

    # Inner lines
    last_group = -1
    for c in range(0, n_columns):
        this_group = cols_subset[c]['group']
        if this_group != last_group:
            last_group = this_group
            x = col_lefts[c+1] - x_eps
            fig.add_artist(matplotlib.lines.Line2D([x, x], [y_bottom, y_top], color='black', linewidth=mid_lines))
    last_group = rows_subset[-1]['group']
    for r in range(1, n_rows):  # From bottom to top
        layout_r = n_rows - r - 1   # ...but the layout is from top to bottom
        this_group = rows_subset[layout_r]['group'] # ...but the layout is from top to bottom
        if this_group != last_group:
            last_group = this_group
            y = row_bottoms[r] - y_eps
            fig.add_artist(matplotlib.lines.Line2D([x_left, x_right], [y, y], color='black', linewidth=mid_lines))
    # ... adds one last inner line for the header
    y = row_bottoms[-1] - y_eps
    fig.add_artist(matplotlib.lines.Line2D([x_left, x_right], [y, y], color='black', linewidth=mid_lines))

    # Outer lines - a bit thicker and longer
    # ... bottom line
    y = y_bottom
    fig.add_artist(matplotlib.lines.Line2D([x_left-x_eps, x_right+x_eps], [y, y], color='black', linewidth=outer_lines))
    # ... top line
    y = y_top
    fig.add_artist(matplotlib.lines.Line2D([x_left-x_eps, x_right+x_eps], [y, y], color='black', linewidth=outer_lines))


add_lines()


# Plotting each cell again
# ... data
def darken(rgb, factor):
    rgb = rgb[:3]
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, l * factor, s)


prefix_colors = layout['prefix_colors']
prefix_darker = [darken(matplotlib.colors.to_rgba(p), 0.5) for p in prefix_colors]


def get_stat(r, c, p):
    try:
        return all_stats[(r, c)][p][STATISTIC]
    except KeyError:
        existing_stats = (k for k, v in all_stats[(r, c)][p].items() if not isinstance(v, list))
        raise ValueError(f'Unknown statistic: {STATISTIC}, try one of: {", ".join(existing_stats)}')


np.random.seed(0)  # Makes the sample clouds consistent across runs

if PLOT_TYPE == 'samples':
    path_effects=[pe.withStroke(linewidth=2, foreground='white')]
else:
    path_effects=[pe.withStroke(linewidth=2, foreground='white')]


OFFSET = 11 
print(f"{len(x_grids)=}")
# 2 rows 
# 11 columns 
for r, row in enumerate(rows_subset):  # Draw from top to bottom, so that the overlapping effect is correct
    for c, col in enumerate(cols_subset):
        ax = get_ax(r+1, c+1) # order change to c+1, r+1 -> before: r+1,
        x_grid = x_grids[r]
        x_min = x_grid.min()
        x_max = x_grid.max()
        x_range = x_max - x_min
        if PLOT_TYPE == 'samples':
            # Add some padding to the x-axis
            ax.set_xlim(x_min - x_range*0.1, x_max + x_range*0.1)
        else:
            ax.set_xlim(x_min, x_max)
        # Sorts the distributions so the means are presented in the right order
        dists = [(p, prefix, get_stat(c, r+OFFSET, p)) for p, prefix in enumerate(col['prefixes']) if prefix is not None]
        p_to_i = {p: i for i, (p, _, _) in enumerate(dists, start=1)}  # Maps prefix to absolute index before sorting
        dists.sort(key=lambda x: x[2])
        dists_n = len(dists)
        assert dists_n <= n_prefixes
        for d, (p, prefix, stat) in enumerate(dists, start=1):
            # Create a faint ridgeline plot in the background
            if PLOT_TYPE == 'hist':
                hist = all_histograms[(c, r)][p] # r,c -> order changed to c, r
                ax.bar(hist[1][:-1], hist[0], width=hist[1][1] - hist[1][0], color=prefix_colors[p], alpha=0.3,
                       clip_on=True)
            elif PLOT_TYPE == 'kde': 
                kde = all_kdes[(c, r+OFFSET)][p] # r,c -> order changed to c, r 
                y = kde.pdf(x_grid)
                color = prefix_colors[0] if r == 0 else prefix_colors[1]
                # ax.fill_between(x_grid, y, color=prefix_colors[p], alpha=0.3, clip_on=True)
                ax.fill_between(x_grid, y, color=color, alpha=0.3, clip_on=True)
            elif PLOT_TYPE == 'samples':
                x = all_subsamples[(r, c)][p]
                # Displaces each distribution a little vertically so to reduce their
                y_m = p_to_i[p] / (dists_n + 1)
                y_sd = 0.25 / (dists_n + 1)
                y = np.random.normal(y_m, scale=y_sd, size=x.shape)
                if not STEADY_SAMPLES:
                    # Adds a tiny amount of jittering. This must be enough to help making the points more visible, but
                    # not enough to influence the data interpretation.
                    x_jitter_std = x_range / 250
                    x_jitter = np.random.normal(0, scale=x_jitter_std, size=x.shape)
                    x += x_jitter
                sp = ax.scatter(x, y, color=prefix_colors[p], alpha=0.3, clip_on=False, linewidths=0, s=1.5)
                sp.set_rasterized(not VECTORIZED_SCATTER)
            else:
                assert False

            # Overlay the mean value
            if d == 1 or summarize_all:
                stat_text = f'{stat * NUMBER_SCALING:{NUMBER_FORMAT}}'
                x_pos = d / (dists_n + 1)
                y_jit = p/dists_n * PREFIX_JITTER
                font_jit = (p-dists_n/2)/dists_n * PREFIX_FONT_JITTER
                fontsize = fontsize_data * (1 + font_jit)
                # fontfamily = PREFIX_FONTS[p % len(PREFIX_FONTS)]
                ax.text(x_pos, 0.5*h_max_row+y_jit, stat_text, horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=fontsize,  # fontfamily=fontfamily,
                        color=prefix_darker[p], path_effects=path_effects)
                d += 1

# ... row headers
for r, row in enumerate(rows_subset, start=1):
    ax = get_ax(r, 0)
    dx = to_axis_width(ax, col_spacing_inches)
    ax.text(1.0-dx, 0.5, row['header'], horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=fontsize)

# ... column headers
for c, col in enumerate(cols_subset, start=1):
    ax = get_ax(0, c)
    ax.text(0.5, 0.5, col['header'], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontsize=fontsize)

if OUTPUT_FILE == '!':
    plt.show()
else:
    plt.savefig(OUTPUT_FILE, bbox_inches='tight')
