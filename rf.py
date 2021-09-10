import numpy as np
import pandas as pd
from july21_get_all_data import get_july21_all_data
from distance_matrix import get_n_groups, calcDTWDist, pearson, spearman
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import itertools
import pickle
import matplotlib.pyplot as plt


def construct_samples(df, cols,col_to_pred,n_steps):
    """
    Turns time series dataframe columns into features and labels for random forest
    :param df: data d
    :param cols: dataframe cols to use
    :param col_to_pred: dataframe column being predicted
    :param n_steps: number of backwards time steps to use
    :return: 
    """
    df = df[cols]
    label_col = df[col_to_pred]
    feature_df = df.drop(columns = col_to_pred)

    f_list = []

    k = len(feature_df)%(n_steps+1)

    x = np.empty(shape = (len(feature_df)-k-n_steps+1,n_steps,len(feature_df.columns)))
    y = np.empty(shape = (len(feature_df)-k-n_steps+1,1))

    for i in range(len(feature_df)-k-n_steps+1):
        x[i] = feature_df.iloc[i:i+n_steps].to_numpy()
        y[i] = label_col.iloc[i+n_steps]

    x = np.array([sa.ravel() for sa in x])
    for i in range(1,n_steps+1):
        f_list.extend([col_name+f'_-{i}' for col_name in feature_df.columns])
    return x,y.ravel(),f_list


def tt_split(x,y,perc):
    """
    splits x and y into train and test datasets 
    :param x: features
    :param y: labels
    :param perc: percentage to keep for training
    :return: x_train,x_test,y_train,y_test
    """
    x_train = x[:int(perc*len(x))]
    x_test = x[int(perc*len(x)):]
    y_train = y[:int(perc * len(y))]
    y_test = y[int(perc * len(y)):]

    return x_train,x_test,y_train,y_test


def get_rf_importances(df,cols,col_to_pred,n_steps,n_trees):
    """
    gets random forest feature importances for predicting one column using other columns in dataframe
    :param df: dataframe
    :param cols: columns to use
    :param col_to_pred: column to predict
    :param n_steps: number of backwards time steps to use when creating features
    :param n_trees: number of trees to use in random forest
    :return: 
    """
    # constructs feature and labels
    x, y, feature_list = construct_samples(df, cols,col_to_pred,n_steps)

    # 80:20 train-test-split on features and labels
    x_train, x_test, y_train, y_test = tt_split(x, y, 0.8)
    
    # defines random forest regressor 
    rf = RandomForestRegressor(n_estimators=n_trees, verbose=0, n_jobs=-1)
    # fits random forest on training data
    rf.fit(x_train, y_train)
    # used random forest to predict test features
    rf_preds = rf.predict(x_test)
    
    # calulcates ev_score, r2, mse, and mape of predictions using test lables
    ev_score = metrics.explained_variance_score(y_test, rf_preds)
    r2 = metrics.r2_score(y_test, rf_preds)
    mse = metrics.mean_squared_error(y_test,rf_preds,squared=True)
    mape = metrics.mean_absolute_percentage_error(y_test,rf_preds)
    
    # extracts feature importances as a dataframe
    rf_feature_importances = pd.DataFrame(rf.feature_importances_,
                                          index=feature_list,
                                          columns=['importance']).sort_values('importance', ascending=False)

    # adds evaluation metrics to dataframe
    rf_feature_importances['mse'] = mse
    rf_feature_importances['r2'] = r2
    rf_feature_importances['ev_score'] = ev_score
    rf_feature_importances['mape'] = mape
    
    # returns dataframe
    return rf_feature_importances


def combi_importances(combi, df, n_steps, n_trees, pbar):
    """
    Returns random forest importances for a combination of columns
    :param combi: combination of columns as list
    :param df: dataframe
    :param n_steps: Number of backwards time steps to use
    :param n_trees: Number of trees to use in random foretse
    :param pbar: tqdm progress bar
    :return: 
    """
    
    i = 0
    combi = np.asarray(combi)
    
    # creates data frame to add importances to
    importance_cols = np.array([[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                                 f'ev_score_{col}', f'mape_{col}'] for col in combi]).ravel()
    all_combi_importances = pd.DataFrame(columns=importance_cols)
    
    # iterates over every column in combination
    for col in combi:
        
        pbar.set_postfix_str(f'{i}/{len(combi)*2}',refresh=True)
        
        # gets importances for predicting col
        importances = get_rf_importances(df, combi, col, n_steps, n_trees)
        
        # sets importance of col predicting col to 0
        for i in range(1, n_steps + 1):
            importances.loc[f"{col}_-{i}", 'importance'] = 0
        
        # fills the evaluation metrics of col
        importances = importances.fillna(importances.mean())
        
        # renames columns
        col_importances = importances.rename(columns={'importance': f"importance_{col}",
                                                      'mse': f"mse_{col}", 'r2': f"r2_{col}",
                                                      'ev_score': f"ev_score_{col}",
                                                      'mape': f"mape_{col}"})
        # adds importances to all_combi_importances
        all_combi_importances[[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                               f'ev_score_{col}', f'mape_{col}']] = col_importances
        i+=1

    return all_combi_importances


def total_norm_scale_sort(imp_df, data_df):
    """
    normalises and scales importances dataframe
    :param imp_df: importances dataframe
    :param data_df: data dataframe
    :return: 
    """
    # scales importance using r2
    for col in data_df.columns:
        # metrics = r2, ev_score, mse, mape
        scale_func = lambda x : x[f'importance_{col}'] * x[f'r2_{col}']
        imp_df = imp_df.assign(new_col = scale_func)
        imp_df = imp_df.rename(columns={'new_col':f'scaled_importance_{col}'})
    
    # sums to get total importance and total scaled importance
    imp_df['total_importance'] = imp_df[[f"importance_{col}" for col in data_df.columns]].sum(axis=1)
    imp_df['total_scaled_importance'] = imp_df[[f"scaled_importance_{col}" for col in data_df.columns]].sum(axis=1)
    
    # normalises total importance
    imp_df['normed_total_importance'] = imp_df['total_importance'] / imp_df['total_importance'].sum()
    
    # normalises total scaled importance
    if imp_df['total_scaled_importance'].min()<0:
        min_val =  imp_df['total_scaled_importance'].min()
        imp_df['normed_total_scaled_importance'] = (imp_df['total_scaled_importance']-min_val) / (imp_df['total_scaled_importance']-min_val).sum()
    else:
        imp_df['normed_total_scaled_importance'] = imp_df['total_scaled_importance'] / imp_df['total_scaled_importance'].sum()

    imp_df = imp_df.sort_values('normed_total_scaled_importance', ascending=False)
    return imp_df


def combi_summed_importances(n_steps,n_trees,combis,df, pickle_dfs = False):
    """
    Get summed importances for each column for a list of column combinations
    :param n_steps: number of backwards time steps to use
    :param n_trees: number of trees to use in random forest
    :param combis: list of column combinations, [combi1, combi2,...,combi_n]
    :param df: dataframe
    :param pickle_dfs: Bool to pickle importances dataframes
    :return: 
    """
    # creates dataframe to add summed importances to
    all_importance_cols = np.array([[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                                     f'ev_score_{col}', f'mape_{col}'] for col in df.columns]).ravel()

    all_importance_index = [f"{col}_-{i}" for i in range(1, n_steps + 1) for col in df.columns]

    all_importances = pd.DataFrame(index=all_importance_index, columns=all_importance_cols)

    all_column_groups_combinations = combis
    
    # iterates over combinations
    with tqdm(all_column_groups_combinations, position=0, leave=True) as pbar:
        i = 0
        for combi in pbar:
            # gets importance for combination
            all_combi_importances = combi_importances(combi, df, n_steps, n_trees, pbar)
            
            # pickles all_combi_importances
            if pickle_dfs:
                pickle.dump(all_combi_importances, open(f"Combi Importances/combi_{i}_imps_{n_steps}_steps_{n_trees}_trees.pkl",
                                                        'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            
            # adds importances to all_importance_cols
            sect = all_importances.columns.intersection(all_combi_importances.columns)
            all_importances[sect] = all_importances[sect].add(all_combi_importances[sect], fill_value=0)
            i+=1
    
    # pickles all_importances
    if pickle_dfs:
        pickle.dump(all_importances, open(f"all_combi_imps_{n_steps}_steps_{n_trees}_trees.pkl", 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        
    # normalises and scales importances
    all_importances = total_norm_scale_sort(all_importances, df)
    
    # gets normed total importance and normed total scaled importance columns 
    all_importances_normed_totals = all_importances[['normed_total_importance', 'normed_total_scaled_importance']]

    # sums importances for each column over backwards time steps
    all_importances_normed_totals_lags_summed = pd.DataFrame(columns=all_importances_normed_totals.columns)
    for col in df.columns:
        col_lag_sum = all_importances.filter(axis=0, regex=f"^{col}_-\d+").sum(axis=0)
        all_importances_normed_totals_lags_summed.loc[col] = col_lag_sum
    all_importances_normed_totals_lags_summed = all_importances_normed_totals_lags_summed.sort_values(
        'normed_total_scaled_importance', ascending=False)

    all_imp_nt = all_importances_normed_totals
    all_imp_nt_ls = all_importances_normed_totals_lags_summed

    return all_imp_nt,all_imp_nt_ls


def col_4_cols_importances(n_steps,n_trees,df):
    """
    Gets rf importances for each col used to predict every other col in df
    :param n_steps: number of backwards timesteps to use
    :param n_trees: number of trees to use in rf
    :param df: dataframe
    :return: 
    """
    # creates dataframe to add importances to 
    importance_cols = np.array([[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                                 f'ev_score_{col}', f'mape_{col}'] for col in df.columns]).ravel()
    all_importances = pd.DataFrame(columns=importance_cols)
    
    # iterates over every column in dataframe
    for col in tqdm(df.columns):
        # gets importances for using all other cols to predict col 
        importances = get_rf_importances(df, df.columns, col, n_steps, n_trees)
        
        # sets importance for col predicting col to 0
        for i in range(1, n_steps + 1):
            importances.loc[f"{col}_-{i}", 'importance'] = 0

        # fills the evaluation metrics of col
        importances = importances.fillna(importances.mean())
        
        # renames columns
        col_importances = importances.rename(columns={'importance': f"importance_{col}",
                                                      'mse': f"mse_{col}", 'r2': f"r2_{col}",
                                                      'ev_score': f"ev_score_{col}",
                                                      'mape': f"mape_{col}"})
        
        # adds importances to all_importances
        all_importances[[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                         f'ev_score_{col}', f'mape_{col}']] = col_importances

    # normalises and scales importances
    all_importances = total_norm_scale_sort(all_importances, df)

    # gets normed total importance and normed total scaled importance columns
    all_importances_normed_totals = all_importances[['normed_total_importance', 'normed_total_scaled_importance']]

    # sums importances for each column over backwards time steps
    all_importances_normed_totals_lags_summed = pd.DataFrame(columns=all_importances_normed_totals.columns)
    for col in df.columns:
        col_lag_sum = all_importances.filter(axis=0, regex=f"^{col}_-\d+").sum(axis=0)
        all_importances_normed_totals_lags_summed.loc[col] = col_lag_sum
    all_importances_normed_totals_lags_summed = all_importances_normed_totals_lags_summed.sort_values(
        'normed_total_scaled_importance', ascending=False)

    all_imp_nt = all_importances_normed_totals
    all_imp_nt_ls = all_importances_normed_totals_lags_summed
    return all_imp_nt, all_imp_nt_ls


def group_sample_importances(all_column_groups, df, n_steps, n_trees):
    """
    Creates a random sample from containing one column from each group in all_column_groups
    :param all_column_groups: list of groups
    :param df: dataframe
    :param n_steps: number of backwards time steps to use
    :param n_trees: number of trees to use in random forest
    :return:
    """
    n_groups = len(all_column_groups)

    # creates dataframe to add sample importances to
    importance_cols = np.array([[f"importance_group_{i}", f'mse_group_{i}', f'r2_group_{i}',
                                 f'ev_score_group_{i}', f'mape_group_{i}'] for i in range(1, n_groups + 1)]).ravel()
    sample_importances = pd.DataFrame(columns=importance_cols)

    # generates random sample
    sample_cols = [np.random.choice(grp) for grp in all_column_groups]

    # iterates over columns in sample and gets importances
    for col in sample_cols:
        # gets importances
        importances = get_rf_importances(df, sample_cols, col, n_steps, n_trees)

        # sets importance for col predicting col to 0
        for i in range(1, n_steps + 1):
            importances.loc[f"{col}_-{i}", 'importance'] = 0

        # fills metric values for col
        importances = importances.fillna(importances.mean())

        # renames columns
        col_importances = importances.rename(columns={'importance': f"importance_group_{sample_cols.index(col) + 1}",
                                                      'mse': f"mse_group_{sample_cols.index(col) + 1}",
                                                      'r2': f"r2_group_{sample_cols.index(col) + 1}",
                                                      'ev_score': f"ev_score_group_{sample_cols.index(col) + 1}",
                                                      'mape': f"mape_group_{sample_cols.index(col) + 1}"})

        # adds col importances to sample importances
        sample_importances[[f"importance_group_{sample_cols.index(col) + 1}", f'mse_group_{sample_cols.index(col) + 1}',
                           f'r2_group_{sample_cols.index(col) + 1}', f'ev_score_group_{sample_cols.index(col) + 1}',
                           f'mape_group_{sample_cols.index(col) + 1}']] = col_importances

    # reindexes sample importances dataframe
    sample_importances_new_index = [f'group_{sample_cols.index(col[:-3]) + 1}{col[-3:]}' for col in
                                   sample_importances.index.values]

    index_mapper = {f'{key}': f'{value}' for (key, value) in
                    zip(sample_importances.index.values, sample_importances_new_index)}

    sample_importances = sample_importances.rename(index=index_mapper)

    return sample_importances


def group_4_groups_importances(all_groups,df,n_steps,n_trees,n_samples):
    """
    gets importances of n_samples random samples and sums importances for each group
    :param all_groups: list of groups
    :param df: dataframe
    :param n_steps: number of backward time steps to use
    :param n_trees: number of trees to use in random forest
    :param n_samples: number of random sample to use
    :return:
    """
    n_groups = len(all_groups)

    # creates dataframe to add group importances to
    importance_index = np.array([f'group_{i}_-{j}' for i in range(1,n_groups+1) for j in range(1,n_steps+1)])
    importance_cols = np.array([[f'normed_total_importance_sample_{i}',f'normed_total_scaled_importance_sample_{i}'] for i in range(1,n_samples+1)]).ravel()
    group_importances = pd.DataFrame(index = importance_index, columns=importance_cols)

    for i in tqdm(range(1,n_samples+1)):
        # importances for random sample
        sample_importances = group_sample_importances(all_groups,df,n_steps,n_trees)

        # scales importances
        for ind in group_importances.index.values:
            group = f'{ind[:-3]}'
            scale_func = lambda x: x[f'importance_{group}'] * x[f'r2_{group}']  # / x[f'mape_{col}']
            sample_importances = sample_importances.assign(new_col=scale_func)
            sample_importances = sample_importances.rename(columns={'new_col': f'scaled_importance_{group}'})

        # sums to get total importance and total scaled importance
        sample_importances['total_importance'] = sample_importances[[f"importance_group_{i}" for i in range(1,n_groups+1)]].sum(axis=1)
        sample_importances['total_scaled_importance'] = sample_importances[[f"scaled_importance_group_{i}" for i in range(1,n_groups+1)]].sum(axis=1)

        # normalises total importance
        sample_importances['normed_total_importance'] = sample_importances['total_importance'] / sample_importances[
                                                                                             'total_importance'].sum()
        # normalises total scaled importance
        if sample_importances['total_scaled_importance'].min() < 0:
            min_val = sample_importances['total_scaled_importance'].min()
            sample_importances['normed_total_scaled_importance'] = \
            (sample_importances['total_scaled_importance'] - min_val) / (sample_importances['total_scaled_importance'] - min_val).sum()
        else:
            sample_importances['normed_total_scaled_importance'] = sample_importances['total_scaled_importance'] / \
                                                                  sample_importances['total_scaled_importance'].sum()

        group_importances[[f'normed_total_importance_sample_{i}',f'normed_total_scaled_importance_sample_{i}']] = \
            sample_importances[['normed_total_importance','normed_total_scaled_importance']]

    # sums sample normed total importance and normed scaled total importance over each sample
    group_importances['normed_total_importance'] = group_importances.filter(axis=1,regex='^normed_total_importance_sample_\d+').mean(axis = 1)
    group_importances['normed_total_scaled_importance'] = group_importances.filter(axis=1, regex='^normed_total_scaled_importance_sample_\d+').mean(axis=1)
    group_importances = group_importances.sort_values('normed_total_scaled_importance', ascending=False)

    # gets normed total importance and normed total scaled importance
    group_importances_nt = group_importances[['normed_total_importance','normed_total_scaled_importance']]

    # sums importances for each column over backwards time steps
    group_importances_nt_ls = pd.DataFrame(columns = group_importances_nt.columns)
    for i in range(1,n_groups+1):
        group_ls = group_importances_nt.filter(axis=0, regex=f"^group_{i}_-\d+").sum(axis=0)
        group_importances_nt_ls.loc[f'group_{i}'] = group_ls

    group_importances_nt_ls = group_importances_nt_ls.sort_values('normed_total_scaled_importance', ascending=False)
        
    return group_importances_nt, group_importances_nt_ls


def group_importance_plot(df, groups, n_steps, n_trees, n_samples, prefix = None):
    """
    Plots bar chart of importance for each group of columns in list of groups
    :param df: dataframe
    :param groups: list of groups
    :param n_steps: number of backwards time steps to use
    :param n_trees: number of trees to use in random forest
    :param n_samples: number of random samples to use
    :param prefix: prefix to print when printing group contents and figure titles
    :return:
    """

    # gets importances dataframes for groups
    gi_nt, gi_nt_ls = group_4_groups_importances(groups, df, n_steps, n_trees, n_samples)

    # prints columns in each group
    i = 1
    for group in groups:
        print(f"{prefix} Group {i} is {' ,'.join(group)}")
        i += 1

    """
    Plots bar chart for importances
    """
    gi_nt.plot.bar()
    plt.ylabel('Forecasting Importance')
    plt.title(f'{prefix}')
    plt.tight_layout()
    plt.show()

    gi_nt_ls.plot.bar()
    plt.ylabel('Forecasting Importance')
    plt.title(f'{prefix}')
    plt.tight_layout()
    plt.show()


def plot_n_groups_importances(df, method, n_groups, n_steps, n_trees, n_samples, use_minThresh = True, use_maxThresh = False):
    """
    Groups df columns into n_groups groups using method then plots bar chart of importances
    :param df: dataframe
    :param method: Func to calculate distance: calcDTWDist(), spearman(), or pearson()
    :param n_groups: number of groups to place columns into
    :param n_steps: number of backwards time steps to use
    :param n_trees: number of trees to use in random forest
    :param n_samples: number of random samples to use
    :param use_minThresh: Bool, plot importances for minimum threshold distance giving n groups
    :param use_maxThresh: Bool, plot importances for maximum threshold distance giving n groups
    :return:
    """

    # gets n_groups using minimum and maximum distance threshold
    min_thresh_groups, max_thresh_groups = get_n_groups(df,method,n_groups)

    # plot figures
    if use_minThresh and use_maxThresh:
        group_importance_plot(df, min_thresh_groups, n_steps, n_trees, n_samples, 'Approach 3 Min Thresh')
        group_importance_plot(df, max_thresh_groups, n_steps, n_trees, n_samples, 'Approach 3 Max Thresh')
    elif use_minThresh:
        group_importance_plot(df, min_thresh_groups, n_steps, n_trees, n_samples, 'Approach 3 Min Thresh')
    elif use_maxThresh:
        group_importance_plot(df, max_thresh_groups, n_steps, n_trees, n_samples, 'Approach 3 Max Thresh')
    else:
        return


def plotColumnImportances(all_imp_nt, all_imp_nt_ls,title):
    """
    Plots column importances bar chart using dataframes
    :param all_imp_nt: normed total importances dataframe
    :param all_imp_nt_ls: normed total lags summed importances dataframe
    :return:
    """
    all_imp_nt.plot.bar()
    plt.ylabel('Forecasting Importance')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    all_imp_nt_ls.plot.bar()
    plt.title(title)
    plt.ylabel('Forecasting Importance')
    plt.tight_layout()
    plt.show()


def plot_cols_4_cols_importances(df, n_steps, n_trees):
    """
    Plots column importances for predicting each column using every other column
    :param df: dataframe
    :param n_steps: number of backward time steps to use
    :param n_trees: number of trees to use in random forest
    :return:
    """
    all_imp_nt, all_imp_nt_ls = col_4_cols_importances(n_steps, n_trees, df)
    plotColumnImportances(all_imp_nt,all_imp_nt_ls,'Approach 1')


def plot_combi_summed_importances(df, combinations,n_steps, n_trees, pickle_dfs = False):
    """
    Plots summed column importances for a list of column combinations
    :param df: dataframe
    :param combinations: list of column combinations
    :param n_steps: number of backward time steps to use
    :param n_trees: number of trees to use in random forest
    :param pickle_dfs: Bool, whether to pickle importances df
    :return:
    """
    all_imp_nt, all_imp_nt_ls = combi_summed_importances(n_steps, n_trees, combinations, df, pickle_dfs)
    if pickle_dfs:
        pickle.dump((all_imp_nt, all_imp_nt_ls), open(f"all_combi_total_imps_{n_steps}_steps_{n_trees}_trees.pkl", 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
    plotColumnImportances(all_imp_nt, all_imp_nt_ls,'Approach 2')


if __name__ == '__main__':
    """
    Plot importances for Approaches 1,2,and 3 using July 21 Hanger Field MIDAS, COSMOS, and flux data
    """
    """
    gets dataframe of july21 data 
    """
    data_df = get_july21_all_data('interp')
    df = data_df

    """
    normalises dataframe
    """
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)
    df = df.reset_index(level=0).drop(columns='index')

    # settings
    n_steps = 2
    n_trees = 100

    """
    Plot Approach 1 importances
    """
    plot_cols_4_cols_importances(df, n_steps, n_trees)


    """
    Manual column groups by sensor type
    """
    temp_columns = ['q10cm_soil_temp','max_air_temp','min_air_temp','min_grss_temp','min_conc_temp','TA_LEVEL2',
                    'STP_TSOIL2_LEVEL2','STP_TSOIL5_LEVEL2', 'STP_TSOIL10_LEVEL2', 'STP_TSOIL20_LEVEL2',
                    'STP_TSOIL50_LEVEL2','TDT1_TSOIL_LEVEL2', 'TDT2_TSOIL_LEVEL2']
    wind_speed_columns = ['mean_wind_speed','max_gust_speed','WS_LEVEL2']
    wind_dir_columns = ['mean_wind_dir','max_gust_dir','WD_LEVEL2']
    rad_columns = ['glbl_irad_amt','LWIN_LEVEL2','SWIN_LEVEL2','LWOUT_LEVEL2','SWOUT_LEVEL2','RN_LEVEL2']
    prcp_columns = ['prcp_amt','PRECIPITATION_LEVEL2']
    humidity_columns = ['Q_LEVEL2','RH_LEVEL2']
    pressure_columns = ['PA_LEVEL2']
    heat_flux_columns = ['H','LE','G1_LEVEL2','G2_LEVEL2']
    soil_moisture_columns = ['COSMOS_VWC','TDT1_VWC_LEVEL2','TDT2_VWC_LEVEL2']
    crns_effective_depth_columns = ['D86_75M']
    co2_flux_columns = ['co2_flux']
    h2o_flux_columns = ['h2o_flux']
    momentum_flux_columns = ['Tau']
    potent_evap_columns = ['PE_LEVEL2_1H']

    all_column_groups = [temp_columns,wind_speed_columns,wind_dir_columns,rad_columns,prcp_columns,humidity_columns,
                         pressure_columns,heat_flux_columns,soil_moisture_columns,crns_effective_depth_columns,
                         co2_flux_columns, h2o_flux_columns,momentum_flux_columns,potent_evap_columns]

    """
    all combinations of 1 column from each group
    """
    all_column_groups_combinations = list(itertools.product(*all_column_groups))

    """
    Plot Approach 2 importances, pickle results
    """
    plot_combi_summed_importances(df, all_column_groups_combinations, n_steps, n_trees, True)

    """
    Plot Approach 3 importances
    """
    # settings
    n_groups = 12
    n_samples = 10
    plot_n_groups_importances(df, calcDTWDist,n_groups,n_steps,n_trees,n_samples,use_minThresh=True, use_maxThresh=True)











