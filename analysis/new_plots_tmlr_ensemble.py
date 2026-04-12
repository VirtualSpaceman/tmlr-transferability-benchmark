# %%

import numpy as np 
import pandas as pd 

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.stats import kendalltau, weightedtau

from itertools import combinations

from utils import normalize_columns, encode_categorical, centralize, bootstrap, grouped_weighted_kendall_tau

# %% [markdown]
# ## Preparing input data 

# %%
print('Preparing data...')
INPUT_FILE = '../inputs/transf_scores.csv'
# INPUT_FILE = '../inputs/transf_scores_14_models.csv'
SAVE_FILE = '../inputs/tmlr_linear_ablations_combinations.csv'
# Read the input data
df_full = pd.read_csv(INPUT_FILE)

# Get both scorers and db informations
ALL_SCORERS = df_full['transf_metric'].unique().tolist()
ALL_DBS =  df_full['dataset'].unique().tolist()
SIGNIFICANCE_LEVEL = 0.05
NUM_MODELS = len(df_full['model'].unique())

# %% [markdown]
# ### Define variables

# %%
DATASET_ORDER = ['caltech101',
                'sun397',
                'voc2007',
                'flowers102',
                'oxfordpets',
                'aircraft',
                'dtd',
                'stanfordcars',
                'brain_tumor_kaggle',
                'breakhis',
                'skin_splits']

SCORER_ORDER = ["ncti_score",        
                "etran_energy_score",
                "pactran_score",     
                "gbc_score",         
                "parc_score",        
                "nleep_score",       
                "leep_score",        
                "logme_score",       
                "tmi_score",         
                "nce_score",         
                "sfda_score",        
                "hscore_score",      
                "reg_hscore_score",  ]


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

LINEAR_ABLATION_NAMES = {'result_overall': 'Pool. completo',
                         'result_pool_per_scorer': 'Pool. per scorer',
                         'result_pool_per_scorer_dataset': 'Pool. per scorer-dataset',
                         'svm': 'SVM', 
                         'least_squares': 'Least Squares',
                         }

DECIMALS = 3

# # Scheme + 3 top + I  
# SCORERS_TO_KEEP = [
#                     'pactran_score',
#                     'etran_energy_score',
#                     'ncti_score',
#                     'imagenet',
#     ]


# %%
def my_bootstrap(x, y, *, groups=None): 
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
        # kendall = bootstrap((x, y), kendall_taus, n_bootstraps=1000)
    else:
        tau, wtau = grouped_kendall_taus(x, y, groups)
        # kendall = bootstrap((x, y, groups), grouped_kendall_taus, n_bootstraps=1000)

    return dict(
            tau=tau,
            wtau=wtau,
            # bootstrapped_tau=[k[0] for k in kendall],
            # bootstrapped_wtau=[k[1] for k in kendall],
        )

# %% [markdown]
# ## Linear models ablations 

# %%
def linear_pooling_overall(df_train, df_test):
    # Linear pooling considering ALL scorers and datasets  
    score = df_train['z_transf_score']
    metric = df_train['z_test_score']
    model = linregress(score, metric)
    # df_test = df_test.head(10) # old 
    df_test = df_test.head(NUM_MODELS) # new 
    # inference
    preds = model.slope * df_test['z_transf_score'] + model.intercept 

    return preds.to_list()

def linear_pooling_per_scorer(df_train, df_test):
    # # Linear pooling per scorer 
    models_dict = dict()
    for scorer_name in df_train['transf_metric'].unique():
        df_scorer_train = df_train[df_train['transf_metric'] == scorer_name]
        score = df_scorer_train['z_transf_score']
        metric = df_scorer_train['z_test_score']
        models_dict[scorer_name] = linregress(score, metric)
    
    # inference -> aggregate by scorer 
    aggregated_predictions = []
    for scorer_name in df_test['transf_metric'].unique():
        df_scorer_test = df_test[df_test['transf_metric'] == scorer_name]
        model = models_dict[scorer_name]
        y_inf = model.slope * df_scorer_test['z_transf_score'] + model.intercept 
        aggregated_predictions.append(np.array(y_inf).reshape(1, -1))
    
    # aggregate predictions 
    aggregated_predictions = np.concatenate(aggregated_predictions).mean(axis=0)
    # transfer_performances = df_test.head(10)['z_test_score'].to_list()
    
    return aggregated_predictions

def linear_pooling_per_score_dataset(df_train, df_test):
    models_dict = dict()
    # train one regression per scorer and dataset
    for dataset_name in df_train['dataset'].unique():
        models_dict[dataset_name] = dict()
        for scorer_name in df_train['transf_metric'].unique():
            df_scorer_train = df_train[(df_train['transf_metric'] == scorer_name) & 
                                       (df_train['dataset'] == dataset_name)]
            score = df_scorer_train['z_transf_score']
            metric = df_scorer_train['z_test_score']
            models_dict[dataset_name][scorer_name] = linregress(score, metric)
    
    # inference 
    aggregated_predictions = []
    # sweep over all datasets in traning and all scorers in test
    for dataset_name in df_train['dataset'].unique():
        for scorer_name in df_test['transf_metric'].unique():
            df_scorer_test = df_test[(df_test['transf_metric'] == scorer_name)] # there will be only 10 measurements             
            # print(df_scorer_test.shape)
            model = models_dict[dataset_name][scorer_name]
            preds = model.slope * df_scorer_test['z_transf_score'] + model.intercept
            aggregated_predictions.append(np.array(preds).reshape(1, -1)) # 1 x 10 
    
    # aggregate predictions 
    aggregated_predictions = np.concatenate(aggregated_predictions).mean(axis=0)
    # transfer_performances = df_test.head(10)['z_test_score'].to_list()
    return aggregated_predictions

# %% [markdown]
# #### SVM Regression 

# %%
def sklearn_regression(df_train, df_test, regression_fn):
    # prepare data for input 
    train_data = dict()
    train_data['z_transf_score'] = list()
    train_data['z_test_score'] = list()
    for idx, scorer in enumerate(SCORERS_TO_KEEP):
        df_scorer = df_train[df_train['transf_metric'] == scorer]
        train_data['z_transf_score'].append(df_scorer['z_transf_score'].to_list())
        
        # Enter here only once to avoid repeated samples
        if idx == 0:
            train_data['z_test_score'].extend(df_scorer['z_test_score'].to_list())

    # concatenate scorers information as features (columns)
    train_data['z_transf_score'] = np.column_stack(train_data['z_transf_score']) # Shape: (N_Archs*N_TrainDatasets, N_Trainscorers)  
    train_data['z_test_score'] = np.array(train_data['z_test_score']) # Shape: (N_Archs*N_TrainDatasets, 1)
    # print(f"{train_data['z_transf_score'].shape=} / {train_data['z_test_score'].shape=}")

    test_data = dict()
    test_data['z_transf_score'] = list()
    test_data['z_test_score'] = list()
    for idx, scorer in enumerate(SCORERS_TO_KEEP):
        df_scorer = df_test[df_test['transf_metric'] == scorer]
        test_data['z_transf_score'].append(df_scorer['z_transf_score'].to_list())
        # Enter here only once to avoid repeated samples
        if idx == 0:
            test_data['z_test_score'].extend(df_scorer['z_test_score'].to_list())

    # concatenate scorers information as features (columns)
    test_data['z_transf_score'] = np.column_stack(test_data['z_transf_score']) # Shape: (N_Archs*N_TestDatasets, N_Trainscorers)  
    test_data['z_test_score'] = np.array(test_data['z_test_score']) # Shape: (N_Archs*N_TestDatasets, 1)
    # print(f"{test_data['z_transf_score'].shape=} / {test_data['z_test_score'].shape=}")

    # train and test SVM for regression 
    regression_fn.fit(X=train_data['z_transf_score'], 
                      y=train_data['z_test_score'])
    predictions = regression_fn.predict(test_data['z_transf_score']) # Shape (N_Archs, 1) 
    # print(f"{predictions.shape=}")
    # kendall_regression = my_bootstrap(predictions, test_data['z_test_score'])
    return predictions


# %%
# Iterate over all available DBS to vary the test DB (leave-one-dataset-out evaluation scheme)

wtau_list_exp = []
NEW_COLUMNS = ['test_db', 'model', 'test_score', 
               'classes', 'pred', 'combination', 
               'method_name']

for num_combinations in range(1, len(SCORER_ORDER)):

    print(f"Processsing all {num_combinations=} combinations  of scorers...")
    for scorer_combination in combinations(SCORER_ORDER, num_combinations):
        # Scheme + 3 top + I  
        SCORERS_TO_KEEP = list(scorer_combination)

        list_summary = []

        for dataset_idx, TEST_DB in enumerate(ALL_DBS, start=1):
            TRAIN_DBS = [db for db in ALL_DBS if db != TEST_DB]

            SELECTED_SCORERS = SCORERS_TO_KEEP.copy()

            DATASETS = sorted(set(TRAIN_DBS) - {TEST_DB}) + [TEST_DB]
            df = df_full[df_full['transf_metric'].isin(SELECTED_SCORERS) & df_full['dataset'].isin(DATASETS)].copy()
            group_by = ['transf_metric', 'dataset']
            columns_to_normalize = ['test_score', 'transf_score']
            normalize_columns(df, columns=columns_to_normalize, group_by=group_by) # in-place normalization 
            
            # Encode the categorical variables as integers
            translation = encode_categorical(df, 
                                            columns=['model', 'transf_metric', 'dataset'], 
                                            encoding = dict( model=None, transf_metric=None, dataset=dict(enumerate(DATASETS, start=1)) ))
            
            # Selects and validates the data
            df_train = df[df['dataset'].isin(TRAIN_DBS)]
            df_test = df[df['dataset'] == TEST_DB]
            if len(df_train) == 0:
                raise ValueError('No training data meets the criteria!')
            if len(df_test) == 0:
                raise ValueError('No test data meets the criteria!')

            df_target_check = df_test.groupby('model')['test_score'].transform(centralize).abs().max()
            if df_target_check > 1e-6:
                raise ValueError('The test scores are not constant for each model on the test dataset!')

            if df_test['classes'].nunique() != 1:
                raise ValueError('The test dataset has different class counts on different entries!')


            # print(f'[{TEST_DB=}][{include_imagenet=}] Fitting models...')
            
            preds_overall = linear_pooling_overall(df_train, df_test)

            preds_per_scorer = linear_pooling_per_scorer(df_train, df_test)

            preds_per_scorer_dataset = linear_pooling_per_score_dataset(df_train, df_test)

            preds_svm = sklearn_regression(df_train, df_test, regression_fn=SVR(C=0.01))

            preds_squares = sklearn_regression(df_train, df_test, regression_fn=LinearRegression())

            y = df_test.head(NUM_MODELS)['z_test_score'].to_list()
        # 
            method_list = [(f'overall_{num_combinations}_scorers', preds_overall),
                        (f'per_scorer_{num_combinations}_scorers', preds_per_scorer),
                        (f'per_scorer_dataset_{num_combinations}_scorers', preds_per_scorer_dataset),
                        (f'svm_{num_combinations}_scorers', preds_svm),
                        (f'lsq_{num_combinations}_scorers', preds_squares)]
            
            for (method_name, method_preds) in method_list:
                for (row_idx, (_, row)) in enumerate(df_test.head(NUM_MODELS).iterrows()):
                    list_summary.append([TEST_DB, 
                                        row['model'], 
                                        row['test_score'], 
                                        row['classes'], 
                                        method_preds[row_idx],
                                        ",".join(SELECTED_SCORERS),
                                        method_name, ])
            # break 
        summary_df = pd.DataFrame(list_summary, 
                                  columns=NEW_COLUMNS)
        all_ablations_methods = summary_df['method_name'].unique()
        for method_name in all_ablations_methods:
            subdf = summary_df[summary_df.method_name == method_name].copy()
            encode_categorical(subdf, columns=['test_db'], first_index=0)

            x = subdf['pred'].values
            y = subdf['test_score'].values
            g = subdf['i_test_db'].values
            stats_exp_dict = my_bootstrap(x, y, groups=g)

            current_combination = list(subdf['combination'].unique())
            if len(current_combination) > 1: 
                continue 
            
            wtau_list_exp.append([method_name, 
                                  ",".join(current_combination), 
                                  stats_exp_dict['wtau']])
        
        # print(wtau_list_exp)
        # break
# %%
summary_df = pd.DataFrame(wtau_list_exp, 
                          columns= ['method_name', 
                                    'combination', 
                                    'wtau'])

# %%
summary_df

# %%
if SAVE_FILE not in [False, None]:
    print(f"Saving file to {SAVE_FILE}")
    summary_df.to_csv(SAVE_FILE, index=False)

# %%



