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
    x_train = x[:int(perc*len(x))]
    x_test = x[int(perc*len(x)):]
    y_train = y[:int(perc * len(y))]
    y_test = y[int(perc * len(y)):]

    return x_train,x_test,y_train,y_test


def get_rf_importances(df,cols,col_to_pred,n_steps,n_trees):
    x, y, feature_list = construct_samples(df, cols,col_to_pred,n_steps)

    x_train, x_test, y_train, y_test = tt_split(x, y, 0.8)

    rf = RandomForestRegressor(n_estimators=n_trees, verbose=0, n_jobs=-1)
    rf.fit(x_train, y_train)

    rf_preds = rf.predict(x_test)

    ev_score = metrics.explained_variance_score(y_test, rf_preds)
    r2 = metrics.r2_score(y_test, rf_preds)
    mse = metrics.mean_squared_error(y_test,rf_preds,squared=True)
    mape = metrics.mean_absolute_percentage_error(y_test,rf_preds)
    '''
    print(f"ev = {ev_score}")
    print(f"r2 = {r2}")
    print(f"rmse = {rmse}")
    '''

    rf_feature_importances = pd.DataFrame(rf.feature_importances_,
                                          index=feature_list,
                                          columns=['importance']).sort_values('importance', ascending=False)

    #rf_feature_importances['importance'] = rf_feature_importances['importance']*ev_score
    #rf_feature_importances['importance'] = rf_feature_importances['importance']/mse
    rf_feature_importances['mse'] = mse
    rf_feature_importances['r2'] = r2
    rf_feature_importances['ev_score'] = ev_score
    rf_feature_importances['mape'] = mape

    return rf_feature_importances
    # with pd.option_context('display.max_rows', None):
    # print("rf", rf_feature_importances)



def combi_importances(combi,df,n_steps, n_trees,pbar):
    i = 0
    combi = np.asarray(combi)

    importance_cols = np.array([[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                                 f'ev_score_{col}', f'mape_{col}'] for col in combi]).ravel()
    all_combi_importances = pd.DataFrame(columns=importance_cols)

    for col in combi:
        pbar.set_postfix_str(f'{i}/{len(combi)*2}',refresh=True)

        importances = get_rf_importances(df, combi, col, n_steps, n_trees)
        for i in range(1, n_steps + 1):
            importances.loc[f"{col}_-{i}", 'importance'] = 0
        importances = importances.fillna(importances.mean())
        col_importances = importances.rename(columns={'importance': f"importance_{col}",
                                                      'mse': f"mse_{col}", 'r2': f"r2_{col}",
                                                      'ev_score': f"ev_score_{col}",
                                                      'mape': f"mape_{col}"})

        all_combi_importances[[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                               f'ev_score_{col}', f'mape_{col}']] = col_importances
        i+=1

    return all_combi_importances


def total_norm_scale_sort(imp_df, data_df):
    for col in data_df.columns:
        # metrics = r2, ev_score, mse, mape
        scale_func = lambda x : x[f'importance_{col}'] * x[f'r2_{col}']# / x[f'mape_{col}']
        imp_df = imp_df.assign(new_col = scale_func)
        imp_df = imp_df.rename(columns={'new_col':f'scaled_importance_{col}'})

    imp_df['total_importance'] = imp_df[[f"importance_{col}" for col in data_df.columns]].sum(axis=1)
    imp_df['total_scaled_importance'] = imp_df[[f"scaled_importance_{col}" for col in data_df.columns]].sum(axis=1)

    imp_df['normed_total_importance'] = imp_df['total_importance'] / imp_df['total_importance'].sum()

    if imp_df['total_scaled_importance'].min()<0:
        min_val =  imp_df['total_scaled_importance'].min()
        imp_df['normed_total_scaled_importance'] = (imp_df['total_scaled_importance']-min_val) / (imp_df['total_scaled_importance']-min_val).sum()
    else:
        imp_df['normed_total_scaled_importance'] = imp_df['total_scaled_importance'] / imp_df['total_scaled_importance'].sum()

    imp_df = imp_df.sort_values('normed_total_scaled_importance', ascending=False)
    return imp_df


def combi_summed_importances(n_steps,n_trees,combis,df):
    all_importance_cols = np.array([[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                                     f'ev_score_{col}', f'mape_{col}'] for col in df.columns]).ravel()

    all_importance_index = [f"{col}_-{i}" for i in range(1, n_steps + 1) for col in df.columns]

    all_importances = pd.DataFrame(index=all_importance_index, columns=all_importance_cols)

    all_column_groups_combinations = combis

    with tqdm(all_column_groups_combinations, position=0, leave=True) as pbar:
        i = 0
        for combi in pbar:
            all_combi_importances = combi_importances(combi, df, n_steps, n_trees, pbar)
            #pickle.dump(all_combi_importances, open(f"Combi Importances/combi_{i}_imps_{n_steps}_steps_{n_trees}_trees.pkl",
                                                    #'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            sect = all_importances.columns.intersection(all_combi_importances.columns)

            all_importances[sect] = all_importances[sect].add(all_combi_importances[sect], fill_value=0)
            i+=1

    #pickle.dump(all_importances, open(f"all_combi_imps_{n_steps}_steps_{n_trees}_trees.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    all_importances = total_norm_scale_sort(all_importances, df)
    all_importances_normed_totals = all_importances[['normed_total_importance', 'normed_total_scaled_importance']]
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
    importance_cols = np.array([[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                                 f'ev_score_{col}', f'mape_{col}'] for col in df.columns]).ravel()
    all_importances = pd.DataFrame(columns=importance_cols)
    for col in tqdm(df.columns):
        importances = get_rf_importances(df, df.columns, col, n_steps, n_trees)
        for i in range(1, n_steps + 1):
            importances.loc[f"{col}_-{i}", 'importance'] = 0
        importances = importances.fillna(importances.mean())
        col_importances = importances.rename(columns={'importance': f"importance_{col}",
                                                      'mse': f"mse_{col}", 'r2': f"r2_{col}",
                                                      'ev_score': f"ev_score_{col}",
                                                      'mape': f"mape_{col}"})

        all_importances[[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                         f'ev_score_{col}', f'mape_{col}']] = col_importances


    all_importances = total_norm_scale_sort(all_importances, df)
    all_importances_normed_totals = all_importances[['normed_total_importance', 'normed_total_scaled_importance']]
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
    n_groups = len(all_column_groups)


    importance_cols = np.array([[f"importance_group_{i}", f'mse_group_{i}', f'r2_group_{i}',
                                 f'ev_score_group_{i}', f'mape_group_{i}'] for i in range(1, n_groups + 1)]).ravel()
    sample_importances = pd.DataFrame(columns=importance_cols)

    sample_cols = [np.random.choice(grp) for grp in all_column_groups]
    for col in sample_cols:

        importances = get_rf_importances(df, sample_cols, col, n_steps, n_trees)
        for i in range(1, n_steps + 1):
            importances.loc[f"{col}_-{i}", 'importance'] = 0
        importances = importances.fillna(importances.mean())
        col_importances = importances.rename(columns={'importance': f"importance_group_{sample_cols.index(col) + 1}",
                                                      'mse': f"mse_group_{sample_cols.index(col) + 1}",
                                                      'r2': f"r2_group_{sample_cols.index(col) + 1}",
                                                      'ev_score': f"ev_score_group_{sample_cols.index(col) + 1}",
                                                      'mape': f"mape_group_{sample_cols.index(col) + 1}"})

        sample_importances[[f"importance_group_{sample_cols.index(col) + 1}", f'mse_group_{sample_cols.index(col) + 1}',
                           f'r2_group_{sample_cols.index(col) + 1}', f'ev_score_group_{sample_cols.index(col) + 1}',
                           f'mape_group_{sample_cols.index(col) + 1}']] = col_importances

    sample_importances_new_index = [f'group_{sample_cols.index(col[:-3]) + 1}{col[-3:]}' for col in
                                   sample_importances.index.values]

    index_mapper = {f'{key}': f'{value}' for (key, value) in
                    zip(sample_importances.index.values, sample_importances_new_index)}


    sample_importances = sample_importances.rename(index=index_mapper)


    return sample_importances


def group_4_groups_importances(all_groups,df,n_steps,n_trees,n_samples):
    n_groups = len(all_groups)



    importance_index = np.array([f'group_{i}_-{j}' for i in range(1,n_groups+1) for j in range(1,n_steps+1)])

    importance_cols = np.array([[f'normed_total_importance_sample_{i}',f'normed_total_scaled_importance_sample_{i}'] for i in range(1,n_samples+1)]).ravel()
    group_importances = pd.DataFrame(index = importance_index, columns=importance_cols)
    
    for i in tqdm(range(1,n_samples+1)):
        sample_importances = group_sample_importances(all_groups,df,n_steps,n_trees)

        for ind in group_importances.index.values:
            group = f'{ind[:-3]}'

            scale_func = lambda x: x[f'importance_{group}'] * x[f'r2_{group}']  # / x[f'mape_{col}']
            sample_importances = sample_importances.assign(new_col=scale_func)
            sample_importances = sample_importances.rename(columns={'new_col': f'scaled_importance_{group}'})
       
        sample_importances['total_importance'] = sample_importances[[f"importance_group_{i}" for i in range(1,n_groups+1)]].sum(axis=1)
        sample_importances['total_scaled_importance'] = sample_importances[[f"scaled_importance_group_{i}" for i in range(1,n_groups+1)]].sum(axis=1)

        sample_importances['normed_total_importance'] = sample_importances['total_importance'] / sample_importances[
            'total_importance'].sum()
        if sample_importances['total_scaled_importance'].min() < 0:
            min_val = sample_importances['total_scaled_importance'].min()
            sample_importances['normed_total_scaled_importance'] = (sample_importances[
                                                                       'total_scaled_importance'] - min_val) / (
                                                                              sample_importances[
                                                                                  'total_scaled_importance'] - min_val).sum()
        else:
            sample_importances['normed_total_scaled_importance'] = sample_importances['total_scaled_importance'] / \
                                                                  sample_importances['total_scaled_importance'].sum()

        group_importances[[f'normed_total_importance_sample_{i}',f'normed_total_scaled_importance_sample_{i}']] = \
            sample_importances[['normed_total_importance','normed_total_scaled_importance']]

    group_importances['normed_total_importance'] = group_importances.filter(axis=1,regex='^normed_total_importance_sample_\d+').mean(axis = 1)
    group_importances['normed_total_scaled_importance'] = group_importances.filter(axis=1, regex='^normed_total_scaled_importance_sample_\d+').mean(axis=1)
    group_importances = group_importances.sort_values('normed_total_scaled_importance', ascending=False)
    group_importances_nt = group_importances[['normed_total_importance','normed_total_scaled_importance']]
    group_importances_nt_ls = pd.DataFrame(columns = group_importances_nt.columns)

    for i in range(1,n_groups+1):
        group_ls = group_importances_nt.filter(axis=0, regex=f"^group_{i}_-\d+").sum(axis=0)
        group_importances_nt_ls.loc[f'group_{i}'] = group_ls

    group_importances_nt_ls = group_importances_nt_ls.sort_values('normed_total_scaled_importance', ascending=False)
        
    return group_importances_nt, group_importances_nt_ls


def group_importance_plot(df, groups, n_steps, n_trees, n_samples, prefix = None):
    gi_nt, gi_nt_ls = group_4_groups_importances(groups, df, n_steps, n_trees, n_samples)
    i = 1

    for group in groups:
        print(f"{prefix} Group {i} is {' ,'.join(group)}")
        i += 1

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


def group_and_get_importances(df, method, n_groups, n_steps, n_trees, n_samples, use_minThresh = True, use_maxThresh = False):
    min_thresh_groups, max_thresh_groups = get_n_groups(df,method,n_groups)

    if use_minThresh and use_maxThresh:
        group_importance_plot(df, min_thresh_groups, n_steps, n_trees, n_samples, 'Min Thresh')
        group_importance_plot(df, max_thresh_groups, n_steps, n_trees, n_samples, 'Max Thresh')
    elif use_minThresh:
        group_importance_plot(df, min_thresh_groups, n_steps, n_trees, n_samples, 'Min Thresh')
    elif use_maxThresh:
        group_importance_plot(df, max_thresh_groups, n_steps, n_trees, n_samples, 'Max Thresh')
    else:
        return











if __name__ == '__main__':


    data_df = get_july21_all_data('interp')
    df = data_df

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)
    df = df.reset_index(level=0).drop(columns='index')

    #print(len(df.columns))

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

    #print(sum([len(a) for a in all_column_groups]))

    n_groups = 5
    n_steps = 2
    n_trees = 5
    n_samples = 10

    group_and_get_importances(df, calcDTWDist,n_groups,n_steps,n_trees,n_samples,use_minThresh=True, use_maxThresh=True)

    """
    gi_nt, gi_nt_ls = group_4_groups_importances(all_column_groups,df,n_steps,n_trees,n_samples)
    #print(gi_nt)
    #print(gi_nt_ls)

    i = 1
    for group in all_column_groups:
        print(f"Group {i} is {' ,'.join(group)}")
        i+=1

    gi_nt.plot.bar()
    plt.ylabel('Forecasting Importance')
    plt.tight_layout()
    plt.show()


    gi_nt_ls.plot.bar()
    plt.ylabel('Forecasting Importance')
    plt.tight_layout()
    plt.show()
    """


    '''
    all_column_groups_combinations = list(itertools.product(*all_column_groups))

    n_steps = 3
    n_trees = 15

    #all_imp_nt, all_imp_nt_ls = combi_summed_importances(n_steps,n_trees,all_column_groups_combinations[:],df)
    all_imp_nt, all_imp_nt_ls = col_4_cols_importances(n_steps,n_trees,df)
    #pickle.dump((all_imp_nt, all_imp_nt_ls), open(f"all_combi_total_imps_{n_steps}_steps_{n_trees}_trees.pkl", 'wb'),
                #protocol=pickle.HIGHEST_PROTOCOL)
    #print(all_imp_nt)
    #print(all_imp_nt_ls)

    all_imp_nt.plot.bar()
    plt.ylabel('Forecasting Importance')
    plt.tight_layout()
    plt.show()

    all_imp_nt_ls.plot.bar()
    plt.ylabel('Forecasting Importance')
    plt.tight_layout()
    plt.show()
    '''









