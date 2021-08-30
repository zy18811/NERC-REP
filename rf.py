import numpy as np
import pandas as pd
from july21_get_all_data import get_july21_all_data
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import re
import matplotlib.pyplot as plt



def construct_samples(df, col_to_pred,n_steps):
    label_col = df[col_to_pred]
    feature_df = df.drop(columns = col_to_pred)
    f_list = []

    k = len(feature_df)%(n_steps+1)

    x = np.empty(shape = (len(feature_df)-k-n_steps+1,n_steps,15))
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


def get_rf_importances(df,col_to_pred,n_steps,n_trees):
    x, y, feature_list = construct_samples(df, col_to_pred,n_steps)

    x_train, x_test, y_train, y_test = tt_split(x, y, 0.8)

    rf = RandomForestRegressor(n_estimators=n_trees, verbose=0, n_jobs=-1)
    rf.fit(x_train, y_train)

    rf_preds = rf.predict(x_test)

    ev_score = metrics.explained_variance_score(y_test, rf_preds)
    #r2 = metrics.r2_score(y_test, rf_preds)
    #rmse = metrics.mean_squared_error(y_test,rf_preds,squared=True)
    '''
    print(f"ev = {ev_score}")
    print(f"r2 = {r2}")
    print(f"rmse = {rmse}")
    '''

    rf_feature_importances = pd.DataFrame(rf.feature_importances_,
                                          index=feature_list,
                                          columns=['importance']).sort_values('importance', ascending=False)

    rf_feature_importances['importance'] = rf_feature_importances['importance']*ev_score
    return rf_feature_importances
    # with pd.option_context('display.max_rows', None):
    # print("rf", rf_feature_importances)

def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None,
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2,
                         value_format.format(h), ha="center",
                         va="center")


if __name__ == '__main__':


    data_df = get_july21_all_data()
    df = data_df

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)
    df = df.reset_index(level=0).drop(columns = 'index')


    n_steps = 10


    all_importances = pd.DataFrame(columns = [f"importance_{col}" for col in df.columns])
    for col in tqdm(df.columns):
        importances = get_rf_importances(df,col,n_steps,50)
        for i in range(1, n_steps + 1):
            importances.loc[f"{col}_-{i}"] = 0
        col_importances = importances.rename(columns = {'importance':f"importance_{col}"})
        all_importances[f"importance_{col}"] = col_importances


    all_importances['total'] = all_importances.sum(axis = 1)

    all_importances['normed_total'] = all_importances['total']/all_importances['total'].sum()
    #all_importances = all_importances.sort_values('normed_total',ascending=False)
    #print(all_importances['normed_total'])

    all_importances_lags_summed = pd.DataFrame(columns = all_importances.columns)
    for col in df.columns:
        col_lag_sum = all_importances.filter(axis = 0, regex = f"{col}_-\d+").sum(axis = 0)
        all_importances_lags_summed.loc[col] = col_lag_sum
    all_importances_lags_summed = all_importances_lags_summed.sort_values('normed_total', ascending=False)

    data = []
    series_labels = []
    for i in range(1,n_steps+1):
        lag_rows = all_importances.filter(axis = 0, regex = f".*_-{i}")['normed_total']
        data.append(lag_rows.to_numpy())
        series_labels.append(f"Lag = -{i}")

    category_labels = [cat.replace(f'_-{n_steps}','') for cat in lag_rows.index.values]

    plot_stacked_bar(data = data, series_labels = series_labels, category_labels = category_labels)
    plt.show()
    '''
    plt.figure()
    plt.bar(all_importances.index.values,all_importances['normed_total'].values)
    plt.xticks(rotation = 25)
    plt.show()
    '''
    '''
    plt.figure()
    plt.bar(all_importances_lags_summed.index.values, all_importances_lags_summed['normed_total'].values)
    plt.xticks(rotation=25)
    plt.show()
    '''






