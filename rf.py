import numpy as np
import pandas as pd
from july21_get_all_data import get_july21_all_data
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

    all_combi_importances_lags_summed = pd.DataFrame(columns=all_combi_importances.columns)
    for col in combi:
        pbar.set_postfix_str(f'{i}/{len(combi) * 2}', refresh=True)
        col_lag_sum = all_combi_importances.filter(axis=0, regex=f"^{col}_-\d+").sum(axis=0)
        all_combi_importances_lags_summed.loc[col] = col_lag_sum
        i+=1

    return (all_combi_importances,all_combi_importances_lags_summed)



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

    all_column_groups_combinations = list(itertools.product(*all_column_groups))
    #print(all_column_groups_combinations)
    #print(len(all_column_groups_combinations))

    n_steps = 3
    n_trees = 1

    all_importance_cols = np.array([[f"importance_{col}", f'mse_{col}', f'r2_{col}',
                                 f'ev_score_{col}', f'mape_{col}'] for col in df.columns]).ravel()

    all_importance_index = [f"{col}_-{i}" for i in range(1,n_steps+1) for col in df.columns]

    all_importances = pd.DataFrame(index = all_importance_index, columns = all_importance_cols)
    all_importances_lags_summed = pd.DataFrame(index = df.columns,columns = all_importance_cols)

    all_column_groups_combinations = all_column_groups_combinations[:]

    with tqdm(all_column_groups_combinations,position=0,leave=True) as pbar:
        for combi in pbar:
            (all_combi_importances, all_combi_importances_lags_summed) = combi_importances(combi,df,n_steps,n_trees,pbar)

            sect = all_importances.columns.intersection(all_combi_importances.columns)

            all_importances[sect] = all_importances[sect].add(all_combi_importances[sect],fill_value = 0)
            all_importances_lags_summed[sect] = all_importances_lags_summed[sect].add(
                all_combi_importances_lags_summed[sect],fill_value = 0)


    all_importances['total_importance'] = all_importances[[f"importance_{col}" for col in df.columns]].sum(axis=1)
    all_importances['total_mse'] = all_importances[[f"mse_{col}" for col in df.columns]].sum(axis=1)
    all_importances['total_r2'] = all_importances[[f"r2_{col}" for col in df.columns]].sum(axis=1)
    all_importances['total_ev_score'] = all_importances[[f"ev_score_{col}" for col in df.columns]].sum(axis=1)
    all_importances['total_mape'] = all_importances[[f"mape_{col}" for col in df.columns]].sum(axis=1)

    all_importances['normed_total_importance'] = all_importances['total_importance'] / all_importances['total_importance'].sum()
    all_importances['normed_total_mape'] = all_importances['total_mape'] / all_importances['total_mape'].sum()

    t_i = all_importances['total_importance']
    n_t_mape = all_importances['normed_total_mape']
    t_mse = all_importances['total_mse']
    t_r2 = all_importances['total_r2']
    t_ev = all_importances['total_ev_score']

    all_importances['scaled_total_importance'] = (t_i * t_ev * t_r2)/ (n_t_mape * t_mse)

    all_importances['normed_scaled_total_importance'] = all_importances['scaled_total_importance'] / all_importances['scaled_total_importance'].sum()
    all_importances = all_importances.sort_values('normed_scaled_total_importance', ascending=False)

    all_importances_lags_summed['total_importance'] = all_importances_lags_summed[[f"importance_{col}" for col in df.columns]].sum(axis=1)
    all_importances_lags_summed['total_mse'] = all_importances_lags_summed[[f"mse_{col}" for col in df.columns]].sum(axis=1)
    all_importances_lags_summed['total_r2'] = all_importances_lags_summed[[f"r2_{col}" for col in df.columns]].sum(axis=1)
    all_importances_lags_summed['total_ev_score'] = all_importances_lags_summed[[f"ev_score_{col}" for col in df.columns]].sum(axis=1)
    all_importances_lags_summed['total_mape'] = all_importances_lags_summed[[f"mape_{col}" for col in df.columns]].sum(axis=1)

    all_importances_lags_summed['normed_total_importance'] = all_importances_lags_summed['total_importance'] / all_importances_lags_summed['total_importance'].sum()
    all_importances_lags_summed['normed_total_mape'] = all_importances_lags_summed['total_mape'] / all_importances_lags_summed['total_mape'].sum()

    t_i_ls = all_importances_lags_summed['total_importance']
    n_t_mape_ls = all_importances_lags_summed['normed_total_mape']
    t_mse_ls = all_importances_lags_summed['total_mse']
    t_r2_ls = all_importances_lags_summed['total_r2']
    t_ev_ls = all_importances_lags_summed['total_ev_score']

    all_importances_lags_summed['scaled_total_importance'] = (t_i_ls)/(t_mse_ls)

    all_importances_lags_summed['normed_scaled_total_importance'] = all_importances_lags_summed['scaled_total_importance'] / all_importances_lags_summed['scaled_total_importance'].sum()
    all_importances_lags_summed = all_importances_lags_summed.sort_values('normed_scaled_total_importance', ascending=False)

    all_imp = all_importances[['normed_total_importance','normed_scaled_total_importance']]
    all_imp_ls = all_importances_lags_summed[['normed_total_importance','normed_scaled_total_importance']]

    #print(all_imp)
    #print(all_imp_ls)

    #plt.figure()
    #plt.bar(all_importances.index.values, all_importances['normed_scaled_total_importance'].values)
    #plt.xticks(rotation=25)
    #all_imp.plot.bar()
    #plt.tight_layout()
    #plt.show()


    #plt.figure()
    #plt.bar(all_importances_lags_summed.index.values, all_importances_lags_summed['normed_scaled_total_importance'].values)
    #plt.xticks(rotation=25)
    all_imp_ls.plot.bar()
    plt.ylabel('Forecasting Importance')
    plt.tight_layout()
    plt.show()







    '''
    n_steps = 3


    all_importances = pd.DataFrame(columns = [f"importance_{col}" for col in df.columns])
    for col in tqdm(df.columns):
        importances = get_rf_importances(df,df.columns,col,n_steps,50)
        for i in range(1, n_steps + 1):
            importances.loc[f"{col}_-{i}"] = 0
        col_importances = importances.rename(columns = {'importance':f"importance_{col}"})
        all_importances[f"importance_{col}"] = col_importances


    all_importances['total'] = all_importances.sum(axis = 1)

    all_importances['normed_total'] = all_importances['total']/all_importances['total'].sum()

    all_importances = all_importances.sort_values('normed_total', ascending=False)

    all_importances_lags_summed = pd.DataFrame(columns = all_importances.columns)
    for col in df.columns:
        col_lag_sum = all_importances.filter(axis = 0, regex = f"^{col}_-\d+").sum(axis = 0)
        all_importances_lags_summed.loc[col] = col_lag_sum
    all_importances_lags_summed = all_importances_lags_summed.sort_values('normed_total', ascending=False)


    plt.figure()
    plt.bar(all_importances.index.values,all_importances['normed_total'].values)
    plt.xticks(rotation = 25)
    plt.show()


    plt.figure()
    plt.bar(all_importances_lags_summed.index.values, all_importances_lags_summed['normed_total'].values)
    plt.xticks(rotation=25)
    plt.show()
    '''






