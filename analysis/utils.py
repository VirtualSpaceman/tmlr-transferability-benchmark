from functools import reduce
from itertools import chain, cycle
import json
import numbers
import os
import re
import sys
import warnings

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.utils import resample  # pylint: disable=import-error


# TODO: accept Sequence instead of tuple in those methods
def get_category_colors():
    '''
    Returns an iterator for a list of colors suitable for categorical data. The iterator cycles through the colors
    after every ~20 iterations.
    '''
    colormap = matplotlib.colormaps['tab20']
    colors = colormap(np.linspace(0, 1, 20))
    colors = [colors[c] for c in chain(range(0, 20, 2), range(1, 20, 2))]
    return cycle(colors)


def needs_update(target, sources):
    '''
    Checks if a target file needs to be updated based on the modification dates of the source files.

    Args:
        target (str): The path of the target file.
        sources (iterable of str): The paths of the source files.

    Returns:
        bool: False if the target file exists and is newer than all source files, True otherwise.
    '''
    if not os.path.exists(target):
        return True
    target_date = os.path.getmtime(target)
    for source in sources:
        if target_date <= os.path.getmtime(source):
            return True
    return False


def bootstrap_ci(data, estimator, *, alpha=0.05, n_bootstraps=1000, packed=False):
    '''
    Classical bootstrapping for confidence intervals.

    Args:
        data (tuple of array-like): The data to bootstrap. Each element of the tuple is a numpy array or pandas series.
            The arrays will be resampled consistently, in parallel, preserving the index-wise correspondences.
        estimator (function): The estimator to apply to the bootstrap samples. The expected signature for estimator is
            results = estimator(*data), where data is the tuple of arrays and results is a number or tuple of numbers.
        alpha (float): The confidence level, given as a tail probability. Defaults to 0.05 (5%).
        n_bootstraps (int): The number of bootstrap samples to draw.
        packed (bool): If True, the results are always returned as a tuple of tuples, even if even if the estimator
            returns a single number.
    Returns:
        tuple or tuple of tuples: The confidence intervals (low, high)  for each element returned by the estimator.
                A single tuple is returned if the estimator returns a single number, otherwise a tuple of tuples is
                returned.
    '''
    bootstrapped = bootstrap(data, estimator, n_bootstraps=n_bootstraps, packed=packed)
    return ci_from_bootstrap(bootstrapped, alpha=alpha, packed=packed)


def bootstrap(data, estimator, *, n_bootstraps=1000, packed=False):
    '''
    Classical bootstrapping raw results.

    Args:
        data (tuple of array-like): The data to bootstrap. Each element of the tuple is a numpy array or pandas series.
            The arrays will be resampled consistently, in parallel, preserving the index-wise correspondences.
        estimator (function): The estimator to apply to the bootstrap samples. The expected signature for estimator is
            results = estimator(*data), where data is the tuple of arrays and results is a number or tuple of numbers.
        n_bootstraps (int): The number of bootstrap samples to draw.
        packed (bool): If True, the results are always returned as a list of tuples, even if the estimator returns a
            single number.
    Returns:
        list of number or list of tuples: Drawn bootstrap results.
    '''
    bootstrapped = []
    results_n = None
    rejected = 0
    repack = len(data) == 1
    for _ in range(n_bootstraps):
        for i in range(10):
            try:
                resampled = resample(*data)
                resampled = (resampled,) if repack else resampled
                results = estimator(*resampled)
            except ValueError as e:
                # This general-purpose exception handling gives a chance to retry the resampling, in case it fails
                # because the resampled data is not valid for the estimator due to random chance.
                rejected += 1
                if i == 9:
                    raise e
                else:
                    continue
            break
        if isinstance(results, tuple):
            if results_n is None:
                results_n = len(results)
            elif len(results) != results_n:
                raise ValueError('estimator must return a tuple of the same length each time.')
        elif packed:
            results = (results,)
            results_n = 1
        bootstrapped.append(results)
    if rejected > 0:
        warnings.warn(f'bootstrap.rejected: {rejected} of {n_bootstraps} bootstrap samples were rejected due to '
                      'ValueErrors in fitting the estimator. A small number of rejections is expected is probably ok, '
                      'but a large number may indicate that the estimator is not robust to the data, and may bias the '
                      'results.')
    return bootstrapped


def ci_from_bootstrap(bootstrapped, *, alpha=0.05, packed=False):
    '''
    Computes the confidence intervals from a list of bootstrap results.

    Args:
        bootstrapped (list of number or list of tuples): Bootstrap results.
        alpha (float): The confidence level, given as a tail probability. Defaults to 0.05 (5%).
        packed (bool): If True, the results are always returned as a tuple of tuples, even if bootstrapped is a list of
            single numbers.

    Returns:
        tuple or tuple of tuples: The confidence intervals (low, high)  for each element returned by the estimator.
                A single tuple is returned if the estimator returns a single number, otherwise a tuple of tuples is
                returned.
    '''
    if not bootstrapped:
        return tuple()
    elif isinstance(bootstrapped[0], tuple):
        packed = True
        results_n = len(bootstrapped[0])
        results = [[x[i] for x in bootstrapped] for i in range(results_n)]
    else:
        results_n = 1
        results = (bootstrapped,)
    intervals = tuple(tuple(np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])) for x in results)
    return intervals if packed else intervals[0]


def deindent(s):
    '''
    Normalizes the indentation of a string by removing the common indentation of the first non-empty line.
    This is useful to write """multi-line strings""" in a readable way, while still being able to use them as input to
    functions where the indentation matters, such as eval() or exec().

    Example:
            deindent("""
                     def f(x):
                         return x + 1
                     """)
            # Returns: 'def f(x):\n    return x + 1\n'

    Caveats: this is intended for space-based indentation. It will probably work for pure tab-based indentation, but
    it may have unexpected results for mixed (tab + space) indentation.
    '''
    lines = s.splitlines()
    indentation = 0
    for line in lines:
        if line.strip():
            indentation = len(line) - len(line.lstrip())
            break
    if indentation == 0:
        return s
    return '\n'.join((line[:indentation].lstrip() + line[indentation:] for line in lines))


def get_regret(x, y):
    '''For a series of (x, y) values, regret = y[best] - y[choice], for best = argmax(y) and choice = argmax(x).'''
    choice = np.argmax(x)
    best = np.argmax(y)
    return y[best] - y[choice]


def normalize(x):
    '''Z-normalizes the pandas series or numpy array x.'''
    return (x - x.mean()) / x.std()


def centralize(x):
    '''Centralizes the pandas series or numpy array x.'''
    return x - x.mean()


def rank_centered(x):
    '''
    Returns the rank of the values in x, centered around 0. The ranks are computed using scipy.stats.rankdata().
    '''
    if isinstance(x, pd.Series):
        x = x.rank()
        x = x - x.mean()
        return x
    else:
        x = np.asarray(x)
        x = rankdata(x)
        x = x - np.mean(x)
    return x


def rank_jittered(x, jitter=0.2):
    '''
    Returns the rank of the values in x, centered around 0. The ranks are computed using scipy.stats.rankdata().
    '''
    return rank_centered(x) + np.random.uniform(-jitter, jitter, len(x))


def kendall_tau(x, y):
    '''
    Computes the Kendall's tau-b correlation coefficient between two arrays x and y. The tau-b coefficient is a
    tie-corrected version of the Kendall's tau coefficient, which is a measure of the correspondence between two
    rankings.
    '''
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = x.size
    if n != y.size:
        raise ValueError('x and y must have the same length')
    if n == 0:
        return np.nan
    N = n * (n - 1) / 2
    x_ties = y_ties = c = d = 0
    for i in range(n):
        for j in range(i+1, n):
            x_sign = np.sign(x[i] - x[j])
            y_sign = np.sign(y[i] - y[j])
            d_sign = x_sign * y_sign
            if d_sign > 0:
                c += 1
            elif d_sign < 0:
                d += 1
            else:
                if x_sign == 0:
                    x_ties += 1
                if y_sign == 0:
                    y_ties += 1
    # print(' K->', N, n*(n-1)/2, x_ties, y_ties, c, d)
    return (c - d) / np.sqrt((N - x_ties) * (N - y_ties))


def invert_permutation(p):
    '''Invert the permutation array p, such that arr[p][invert_permutation(p)] == arr.'''
    p = np.asanyarray(p)
    p_inverse = np.empty_like(p)
    p_inverse[p] = np.arange(p.size)
    return p_inverse


def weighted_kendall_tau(x, y, weighter=None):
    '''
    Computes the weighted Kendall's tau-b correlation coefficient between two arrays x and y. Reverts to the
    unweighted version if weighter is lambda r: 1. By default, the weighter is lambda r: 1/(1+r), which is known as
    the hyperbolic weighter.
    '''
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = x.size
    if n != y.size:
        raise ValueError('x and y must have the same length')
    if n == 0:
        return np.nan
    if weighter is None:
        weighter = lambda r: 1./(1+r)
    x_ranks = np.lexsort((y, x))  # Sort by x, then by y
    x_ranks = x_ranks[::-1]
    x_ranks = invert_permutation(x_ranks)
    y_ranks = np.lexsort((x, y))  # Sort by y, then by x
    y_ranks = y_ranks[::-1]
    y_ranks = invert_permutation(y_ranks)
    x_ties = y_ties = c = d = 0
    # j_ties = 0
    W = 0
    for i in range(n):
        for j in range(i+1, n):
            x_sign = np.sign(x[i] - x[j])
            y_sign = np.sign(y[i] - y[j])
            d_sign = x_sign * y_sign
            wx = weighter(x_ranks[i]) + weighter(x_ranks[j])
            wy = weighter(y_ranks[i]) + weighter(y_ranks[j])
            w = (wx + wy)/4
            # w = weighter(min(x_ranks[i], x_ranks[j]))
            if d_sign > 0:
                c += w
            elif d_sign < 0:
                d += w
            else:
                if x_sign == 0:
                    x_ties += w
                if y_sign == 0:
                    y_ties += w
                # if x_sign == 0 and y_sign != 0:
                #     x_ties += w
                # elif y_sign == 0 and x_sign != 0:
                #     y_ties += w
                # else: #  y_sign == 0 and x_sign == 0:
                #     j_ties += w
            W += w
    # print('WK->', W, n*(n-1)/2, x_ties, y_ties, c, d)
    # Note this is exactly the same formula used in scipy.stats.weightedtau, but there will be numerical differences
    # because scipy uses the O(n log n)-optimized algorithm by Vigna ("A Weighted Correlation Coefficient for Ranking
    # with Trees"), where C-D is computed as W - (x_ties + y_ties - j_ties) + 2*d (formula just below Eq. 4)
    return (c - d) / (np.sqrt((W - x_ties) * (W - y_ties)))
    # return (W - (x_ties + y_ties - j_ties) - 2*d) / (np.sqrt((W - x_ties) * (W - y_ties)))


def grouped_weighted_kendall_tau(x, y, g, weighter=None):
    '''
    Computes the grouped weighted Kendall's tau-b correlation coefficient between two arrays x and y. Reverts to the
    simple weighted version if weighter all samples are in the same group. The groups are defined by the array g.
    The grouped Kendall counts the number of concordant and discordant pairs WITHIN each group, and then computes the
    global weighted Kendall's tau-b correlation coefficient considering ALL pairs.
    '''
    # Prepares the data
    all_x = np.asarray(x).ravel()
    all_y = np.asarray(y).ravel()
    all_g = np.asarray(g).ravel()
    n = all_x.size
    if n != all_y.size:
        raise ValueError('x and y must have the same length')
    if n != all_g.size:
        raise ValueError('x and g must have the same length')
    if weighter is None:
        weighter = lambda r: 1./(1+r)
    # Processes the groups, accumulating the statistics for all of them
    x_ties = y_ties = c = d = 0
    W = 0
    groups = np.unique(all_g)
    for g in groups:
        # Selects within-group data
        group_mask = all_g == g
        x = all_x[group_mask]
        y = all_y[group_mask]
        n = x.size
        assert n > 0  # It should not be possible to have an empty group
        # Finds within-group statistics
        x_ranks = np.lexsort((y, x))  # Sort by x, then by y
        x_ranks = x_ranks[::-1]
        x_ranks = invert_permutation(x_ranks)
        y_ranks = np.lexsort((x, y))  # Sort by y, then by x
        y_ranks = y_ranks[::-1]
        y_ranks = invert_permutation(y_ranks)
        for i in range(n):
            for j in range(i+1, n):
                x_sign = np.sign(x[i] - x[j])
                y_sign = np.sign(y[i] - y[j])
                d_sign = x_sign * y_sign
                wx = weighter(x_ranks[i]) + weighter(x_ranks[j])
                wy = weighter(y_ranks[i]) + weighter(y_ranks[j])
                w = wx + wy
                # w = weighter(min(x_ranks[i], x_ranks[j]))
                if d_sign > 0:
                    c += w
                elif d_sign < 0:
                    d += w
                else:
                    if x_sign == 0:
                        x_ties += w
                    if y_sign == 0:
                        y_ties += w
                W += w
    # Computes the tau based on the accumulated statistics for all groups
    return (c - d) / (np.sqrt((W - x_ties) * (W - y_ties)))


def normalize_columns(df, columns, *, group_by, function=normalize, prefix='z_'):
    '''
    Z-normalizes the columns of a dataframe within the groups defined by the list of columns. The modifications are done
    in-place.

    Args:
        df (pandas.DataFrame): The dataframe to encode.
        columns (list): The columns with data to z-normalize. Each will result in a new column prefixed with 'z_'.
        group_by (list): The columns to group by.
        function (function): The function to apply to the columns. Defaults to normalize().

    Returns:
        pandas.DataFrame: The same `df` dataframe, with the new columns added.
    '''
    for column in columns:
        df[f'{prefix}{column}'] = df.groupby(group_by)[column].transform(function)
    return df


def encode_categorical(df, columns, *, first_index=1, encoding=None):
    '''
    Encodes categorical variables in a dataframe as integers. The modifications are done in-place.

    Args:
        df (pandas.DataFrame): The dataframe to encode.
        columns (list): The columns with categorical data to encode. Each column will result in a new integer column
            prefixed with 'i_'.
        first_index (int): The index of the first integer category. Defaults to 1 (as required by Stan) to encode the
            categories from 1 to N.
        encoding (dict of str -> str -> int or None): A dictionary with the encoding for each column. Each column should
            map to a dict str -> int with the encoding for that column, or to None to use the default encoding.

    Returns:
        dict of dicts: A dictionary with the translation from the integer categories to the original categories:
                       column -> new category index -> old category value
    '''
    encoding = {} if encoding is None else dict(encoding)
    translation = {}
    # Stan is 1-indexed, while Pandas and Python are 0-indexed
    for factor in columns:
        if encoding and encoding[factor] is not None:
            translation[factor] = encoding[factor]
            reversed_map = {v: k for k, v in encoding[factor].items()}
            df[f'i_{factor}'] = df[factor].map(reversed_map.get)
        else:
            factorization = pd.factorize(df[factor], sort=True)
            df[f'i_{factor}'] = factorization[0] + first_index
            translation[factor] = dict(enumerate(factorization[1], start=first_index))
    return translation


def print_full_df(df, *, file=None):
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 4,
                           'display.width', int(1e6)):
        print(df, file=file)


def arviz_diagnostics(posterior, *, plot_path=None, file=None, **summary_kwargs):
    '''
    Analyses default diagnostics from ArviZ: parameter statistics, effective sample size, and R-hat.

    Args:
        posterior (stan.fit): The posterior from a Stan model.
        plot_path (str): If informed, additional detailed plots for the MCMC chains are saved to this file. The suffix
            of the file determines the format of the file (e.g., .pdf, .png, .svg).
        file (file-like object): The file to write the output to. If None, the output is written to stdout.
    '''
    # Prints posterior diagnostics
    idata = az.from_pystan(posterior=posterior)
    summary_kwargs.setdefault('round_to', 4)
    summary = az.summary(idata, **summary_kwargs)
    print_full_df(summary, file=file)
    # Plots diagnostics to file
    if plot_path:
        az.plot_trace(idata, compact=True)
        plt.savefig(plot_path, bbox_inches='tight')
    return summary


def arviz_worst_cases(posterior, arviz_summary):
    '''
    Computes the worst-case values for the R-hat and effective sample size (ESS) statistics from ArviZ, considering all
    estimated parameters.

    Args:
        posterior (stan.fit): The posterior from a Stan model.
        arviz_summary (pandas.DataFrame): The summary of the posterior from arviz.summary().

    Returns:
        dict: The worst-case values for the 'r_hat', 'ess_bulk', and 'ess_tail' statistics, the latter two normalized
            by the number of samples.
    '''
    n_samples = posterior.num_samples * posterior.num_chains
    r_hat = (arviz_summary['r_hat']-1.0).abs().max()
    ess_bulk = arviz_summary['ess_bulk'].min() / n_samples
    ess_tail = arviz_summary['ess_tail'].min() / n_samples
    return {'r_hat': r_hat, 'ess_bulk': ess_bulk, 'ess_tail': ess_tail}


def clean_posterior_df(df):
    '''
    Cleans the output of a Stan model by removing the raw parameters and Stan's internal parameters.
    '''
    df = df.loc[:, ~df.columns.str.endswith('_raw')]
    df = df.loc[:, ~df.columns.str.contains('_raw_')]
    df = df.loc[:, ~df.columns.str.contains('_raw.')]
    df = df.loc[:, ~df.columns.str.endswith('__')]
    df = df.loc[:, ~df.columns.str.startswith('_')]
    return df


def correlation_diagnostics(df, *, threshold=0.8, file=None):
    '''
    Analyses the output of a Stan model for excessive correlations between
    parameters.

    Args:
        df (pandas.DataFrame): The output of a Stan model.
        threshold (float): The threshold for the correlation coefficient.
        file (file-like object): The file to write the output to. If None, the
            output is written to stdout.
    '''
    df = clean_posterior_df(df)
    corr_matrix = df.corr()
    for i, column in enumerate(corr_matrix.columns):
        # Iterate over the upper triangle of the matrix
        for _j, row in enumerate(corr_matrix.index[i + 1:], i + 1):
            if abs(corr_matrix.at[row, column]) > threshold:
                print(f"{column} and {row} have a correlation of {corr_matrix.at[row, column]:.2f}", file=file)


class ToListEncoder(json.JSONEncoder):
    '''Encodes numpy arrays and pandas series as lists when serializing to JSON.'''
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, pd.Series):
            return o.tolist()
        elif isinstance(o, numbers.Number):
            # Convert numpy numbers to Python numbers
            if isinstance(o, numbers.Integral):
                return int(o)
            elif isinstance(o, numbers.Real):
                return float(o)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


preprocessor_name = re.compile(r'[\w_][\w\d_]*')
preprocessor_macro = re.compile(r'\${(\w+)}')


def _replace_macros(input_string, macros):
    '''
    Replaces all occurrences of ${<name>} in a string with the corresponding value in a dictionary.
    '''
    def replace_in_dict(match):
        macro_name = match.group(1)
        return str(macros[macro_name])
    return preprocessor_macro.sub(replace_in_dict, input_string)


def conditional_preprocessor(input_source, defines=None, keep_lines=True):
    '''
    Implements a simple C-like preprocessor. The preprocessor supports the following directives:

    #define <name> - defines a macro with the default value 'True'
    #define <name> <value> - defines a macro with a value
    #undef <name> - undefines a macro
    #ifdef <name> - includes the following lines if <name> is defined
    #ifndef <name> - includes the following lines if <name> is not defined
    #ifalldef <name1> <name2> ... - includes the following lines if all the <name>s are defined
    #ifanydef <name1> <name2> ... - includes the following lines if any of the <name>s is defined
    #ifnonedef <name1> <name2> ... - includes the following lines if none of the <name>s is defined
    #ifonedef <name1> <name2> ... - includes the following lines if one and only one of the <name>s is defined
    #else - includes the following lines if the previous #ifdef or #ifndef was not included
    #endif - closes the previous #ifdef or #ifndef

    #warning - prints a warning message to stderr
    #error - prints an error message to stderr and raises a ValueError

    The preprocessor supports macro-substitution with ${<name>}. The substitution is performed anywhere
    in the line, including inside comments and strings.

    Conditionals may be nested.

    Args:
        input_source (str): The source code to preprocess.
        defines (dict of str -> str): The macros that are defined before the input_source is processed. Defaults
            to an empty dictionary.
        keep_lines (bool): If True, the lines that are skipped by the preprocessor are replaced by empty lines,
            helping to keep the line numbers in the output the same as in the input. If False, the skipped lines are
            removed. Defaults to True.

    Returns:
        str: The preprocessed source code.
    '''
    defines = dict(defines) if defines else dict()
    output_lines = []
    include_stack = []
    include = True

    binary_directives = {'ifdef', 'ifndef', 'undef'}
    manyary_directives = {'ifanydef', 'ifalldef', 'ifnonedef', 'ifonedef', 'info', 'warning', 'error'}
    # unary_directives = {'else', 'endif'}

    for line_i, line in enumerate(input_source.splitlines()):
        stripped_line = line.strip()
        tokens = stripped_line.split()

        if tokens and tokens[0].startswith('#'):
            directive = tokens[0][1:]
            if directive in binary_directives and len(tokens) != 2:
                raise ValueError(f'Line {line_i}: invalid {tokens[0]}, expected "{tokens[0]} <name>", found: '
                                 f'"{line}"')
            elif directive in manyary_directives and len(tokens) < 2:
                raise ValueError(f'Line {line_i}: invalid {tokens[0]}, expected "{tokens[0]} <names...>", found: '
                                 f'"{line}"')
            # Tokens after unary directives are silently ignored to allow closing comments
            match directive:
                case 'define':
                    if include:
                        if preprocessor_name.fullmatch(tokens[1]) is None:
                            raise ValueError(f'Line {line_i}: invalid macro name: "{tokens[1]}"')
                        # TODO: enhance this - the code below has the bug of including the name of the macro
                        # value_p = line.find('#define') + len('#define')
                        # value = line[value_p:].strip()
                        # defines[tokens[1]] = _replace_macros(value, defines) if value else 'True'
                        defines[tokens[1]] = tokens[2] if len(tokens) > 2 else 'True'
                case 'undef':
                    if include:
                        if preprocessor_name.fullmatch(tokens[1]) is None:
                            raise ValueError(f'Line {line_i}: invalid macro name: "{tokens[1]}"')
                        defines.pop(tokens[1], None)
                case 'ifdef':
                    include_stack.append(tokens[1] in defines)
                case 'ifndef':
                    include_stack.append(tokens[1] not in defines)
                case 'ifanydef':
                    include_stack.append(any(token in defines for token in tokens[1:]))
                case 'ifalldef':
                    include_stack.append(all(token in defines for token in tokens[1:]))
                case 'ifnonedef':
                    include_stack.append(all(token not in defines for token in tokens[1:]))
                case 'ifonedef':
                    include_stack.append(sum(token in defines for token in tokens[1:]) == 1)
                case 'else':
                    if include_stack:
                        include_stack[-1] = not include_stack[-1]
                    else:
                        raise ValueError(f'Line {line_i}: #else without matching #ifdef or #ifndef')
                case 'endif':
                    if include_stack:
                        include_stack.pop()
                    else:
                        raise ValueError(f'Line {line_i}: #endif without matching #ifdef or #ifndef')
                case 'info':
                    if include:
                        message_p = line.find('#info') + len('#info')
                        message = line[message_p:].strip()
                        print(f'INFO (line {line_i}): {message}', file=sys.stderr)
                case 'warning':
                    if include:
                        message_p = line.find('#warning') + len('#warning')
                        message = line[message_p:].strip()
                        print(f'WARNING (line {line_i}): {message}', file=sys.stderr)
                case 'error':
                    if include:
                        message_p = line.find('#error') + len('#error')
                        message = line[message_p:].strip()
                        print(f'ERROR (line {line_i}): {message}', file=sys.stderr)
                        raise ValueError(f'Error directive on line {line_i}: {message}')
                case _:
                    raise ValueError(f'Line {line_i}: unknown preprocessor directive: "{line}"')
            include = reduce(lambda x, y: x and y, include_stack, True)
            if keep_lines:
                output_lines.append('//' + line)
        else:
            # Check if current line should be included based on the conditions
            if include:
                output_lines.append(_replace_macros(line, defines))
            elif keep_lines:
                output_lines.append('')

    if include_stack:
        raise ValueError(f'Line {line_i}: missing #endif at the end of the source')  # pylint: disable=undefined-loop-variable

    return '\n'.join(output_lines)
