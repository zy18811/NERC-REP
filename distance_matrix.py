"""
Functions for calculating distance matrix and grouping by a threshold distance for dataframe columns
"""

import numpy as np
import pandas as pd
from dtaidistance import dtw
from tqdm import tqdm
from july21_get_all_data import get_july21_all_data
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from scipy.stats import spearmanr, pearsonr

# value to be treated as zero
zeroValue = 1*(10**-3)


def calcDTWDist(df,col1,col2):
    """
    Calculates the dtw euclidean distance using the fastdtw library
    :param df:
    :param col1:
    :param col2:
    :return:
    """
    distance = dtw.distance_fast(df[col1].to_numpy(), df[col2].to_numpy())
    return distance


def spearman(df,col1,col2):
    """
    Calculates the spearman distance, 0 -> perfect correlation, 1 -> no correlation
    :param df:
    :param col1:
    :param col2:
    :return:
    """
    res = spearmanr(df[col1].to_numpy(),df[col2].to_numpy())
    mod_res = 1-abs(res[0])

    return mod_res


def pearson(df,col1,col2):
    """
    Calculates the pearson distance, 0 -> perfect correlation, 1 -> no correlation
    :param df:
    :param col1:
    :param col2:
    :return:
    """
    res = pearsonr(df[col1].to_numpy(), df[col2].to_numpy())
    mod_res = 1-abs(res[0])

    return mod_res

def doDistanceCalc(index, currentCol, df, distFunc):
    """
    Finds distances between currentCol and other cols in df using distFunc
    :param index: index of currentCol
    :param currentCol: name of col
    :param df: dataframe
    :param distFunc: calcDTWDist, pearson, or spearman
    :return:
    """

    currentRow = []
    cols=df.columns[:index]     # Columns we need to compare too, changes each index to not repeat combinations
    for col in cols:
        currentRow.append(distFunc(df,currentCol,col))
    currentRow.append(0)     # After len(cols), will be the comparison to itself, i.e. 0 distance

    for n in range(len(df.columns)-1-index):
        currentRow.append(np.nan)   # Remaining values in row can be nans as will be repeats
    return currentRow


def distMatrix(df,distFunc):
    """
    Returns distance matrix for columns in df
    :param df: dataframe
    :param distFunc: function to calulcate distance
    :return: dictionary: {'fullList':column names, 'fullMatrix':distance matrix}
    """
    results = dict()
    results["fullList"] = df.columns

    args = [(i, col, df) for i, col in enumerate(df.columns)]

    # Do each column seperately
    dtwRowList = []
    for a in tqdm(args):
        dtwRowList.append(doDistanceCalc(*a, distFunc))
    #for row in dtwRowList:print(row)
    results["fullMatrix"] = np.array(dtwRowList)
    return results


def compressMatrixValues(matrix,colNames,threshold):
    """
    Given a matrix of values corresponding to how similar 2 features are, returns a list of groups of features that
    has up to the same similari
    :param matrix: Numpy matrix
    :param colNames: The sensor headings ordered the same as the indexes in the matrix
    :param threshold:
    :return: A list of sets of features that have distance less than the threshold
    """

    def merge(lsts):
        """
        Merges lists with non-zero intersection of elements, returns sets of elements that were merged
        :param lsts:
        :return:
        """
        sets = [set(lst) for lst in lsts if lst]
        merged = True
        while merged:
            merged = False
            results = []
            while sets:
                common, rest = sets[0], sets[1:]
                sets = []
                for x in rest:
                    if x.isdisjoint(common):
                        sets.append(x)
                    else:
                        merged = True
                        common |= x
                results.append(common)
            sets = results
        return sets

    rowIndex,colIndex = np.where(matrix<=threshold)
    indexes = list(zip(rowIndex,colIndex))
    # List of pairs of sensor names that have distance less than the threshold
    groupedFeaturesList = [(colNames[featureIndex1],colNames[featureIndex2]) for featureIndex1, featureIndex2 in indexes]

    return merge(groupedFeaturesList)


def graphMatrixDistribution(matrix,title = None):
    """
    Plots bar chart of distribution of distances in distance matrix
    :param matrix: distance matrix
    :param title: title of figure
    :return:
    """
    values = matrix.tolist()
    values = np.delete(values, np.where(values==0))
    values = np.delete(values, np.where(np.isnan(values)))
    plt.figure()
    if title!=None:
        plt.title(title)
    plt.hist(values,bins = 'auto',ec = 'black')
    plt.xlabel("Distance Value")
    plt.ylabel("Frequency")
    plt.show()


def graphMatrixHeatmap(matrix, title = None,palette = 'Spectral_r'):
    """
    Plots heatmap of distance matrix
    :param matrix: distance matric
    :param title: title of figure
    :param palette: seaborn color palette to use
    :return:
    """
    labels = matrix['fullList'].values

    sns.set_theme(style="white")

    # removes upper triangle and it is a duplicate of lower triangle
    mask = np.triu(np.ones_like(matrix['fullMatrix'], dtype=bool), 1)

    # turns palette into heatmap
    cmap = sns.color_palette(palette, as_cmap=True)

    # plots heatmap
    sns.heatmap(matrix['fullMatrix'], mask=mask, cmap=cmap, center=0, xticklabels=labels, yticklabels=labels,
                square=True, linewidths=.5, cbar_kws={"shrink": .5, 'label': 'Distance Value'})

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()


def getThresholdandNumFeaturesLeft(matrix):
    """
    For some matrix of similarity values, calculates the number of groups of features for different threshold values
    :param matrix:
    :return:
    """
    colNames = [n for n in range(len(matrix))]

    uniqueVals = np.delete(np.unique(matrix),np.where(np.isnan(np.unique(matrix))))

    # Caps it to up to 1000 thresholding values, equally spaced indexes
    thresholdIndexes = np.unique(np.linspace(zeroValue,len(uniqueVals)-1,1000).astype(int))

    thresholdValues = np.array([uniqueVals[index] for index in thresholdIndexes])

    numOfGroups = []

    for threshold in tqdm(thresholdValues,desc="Threshold Num Groups"):
        numOfGroups.append(len(compressMatrixValues(matrix=matrix,colNames=colNames,threshold=threshold)))

    return thresholdValues,numOfGroups


def findKneePoint(xs,ys):
    """
    Finds knee point of plot
    :param xs: x values
    :param ys: y values
    :return:
    """
    kn = KneeLocator(xs, ys, curve='convex', direction='decreasing',online=True,interp_method='polynomial',
                     polynomial_degree=7)
    return kn.knee


def graphThresholdValues(xs,ys,kneePoint,title = None):
    """
    Plots number of groups against threshold values
    :param xs: x values
    :param ys: y values
    :param kneePoint: knee point
    :param title: figure title
    :return:
    """
    plt.figure()

    plt.vlines(kneePoint,min(ys),max(ys), linestyles='dashed',label = 'Knee Point')

    plt.plot(xs, ys)

    if title is not None:
        plt.title(title)

    plt.xlabel(f"Threshold values, raw knee at {kneePoint:.2f}")
    plt.ylabel(f"Number of grouped features, raw knee at {(ys[list(xs).index(kneePoint)]):.2f}")
    plt.legend()
    plt.show()


def graphFeatureGroupsSubplots(df,featureGrouping,title = None):
    """
    Plots each group as a subplot
    :param df: dataframe
    :param featureGrouping: list of groups [group1, group2,... group_n]
    :param title: figure title
    :return:
    """
    plt.figure()

    if title is not None:
        plt.suptitle(title)

    count = 1
    for group in featureGrouping:
        plt.subplot(len(featureGrouping), 1, count)
        for col in group:
            plt.plot(df[col].index.values, df[col].values, label=col)
        plt.legend()
        count += 1
    plt.tight_layout()
    plt.show()


def getFeatureGrouping(df,method,kn = None):
    """
    Group dataframe columns by distance
    :param df: dataframe
    :param method: Method to calculate distance, 'dtw', 'spearman', or 'pearson'
    :param kn: knee point
    :return: grouped columns or -1 if method was invalid
    """
    if method == 'dtw':
        m = calcDTWDist
    elif method == 'spearman':
        m = spearman
    elif method == 'pearson':
        m = pearson
    else:
        return -1

    # calculates distance matrix
    mat = distMatrix(df,m)

    if kn is None:
        xs, ys = getThresholdandNumFeaturesLeft(mat['fullMatrix'])
        kn = findKneePoint(xs, ys)

    featureGrouping = compressMatrixValues(mat['fullMatrix'], mat['fullList'], kn)
    return featureGrouping


def get_n_groups(df,method,n):
    """
    Group dataframe columns into n groups
    :param df: dataframe
    :param method: Method to calculate distance, 'dtw', 'spearman', or 'pearson'
    :param n: Number of groups
    :return: n groups for min and max threshold value
    """
    # calculates distance matrix
    mat = distMatrix(df, method)

    # gets min and max threshold values for n groups
    xs, ys = getThresholdandNumFeaturesLeft(mat['fullMatrix'])
    xs = np.array(xs)
    ys = np.array(ys)
    n_kns = xs[np.where(ys == n)]
    min_thresh = n_kns.min()
    max_thresh = n_kns.max()

    # gets groups for min threshold value
    min_groups = compressMatrixValues(mat['fullMatrix'],mat['fullList'],min_thresh)
    min_groups = [list(g) for g in min_groups]

    # gets groups for max threshold value
    max_groups = compressMatrixValues(mat['fullMatrix'],mat['fullList'],max_thresh)
    max_groups = [list(g) for g in max_groups]

    return min_groups,max_groups


if __name__ == '__main__':
    """
    Plots all figures using all 3 methods for Spen Farm, Hanger Field, July 2021 MIDAS, COSMOS, and flux data 
    """
    data_df = get_july21_all_data('interp')
    df = data_df

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)

    dtw_mat = distMatrix(df,calcDTWDist)
    pear_mat = distMatrix(df,pearson)
    spr_mat = distMatrix(df,spearman)

    # dtw
    graphMatrixDistribution(dtw_mat['fullMatrix'],title='DTW')
    graphMatrixHeatmap(dtw_mat, title = 'DTW')
    xs, ys = getThresholdandNumFeaturesLeft(dtw_mat['fullMatrix'])
    kn = findKneePoint(xs, ys)
    graphThresholdValues(xs, ys, kneePoint=kn,title='DTW')
    featureGrouping = compressMatrixValues(dtw_mat['fullMatrix'], dtw_mat['fullList'], 2.7)
    print('DTW',featureGrouping)
    graphFeatureGroupsSubplots(df,featureGrouping,title = 'DTW')

    # pearson
    graphMatrixDistribution(pear_mat['fullMatrix'], title='Pearson')
    graphMatrixHeatmap(pear_mat,title = 'Pearson')
    xs, ys = getThresholdandNumFeaturesLeft(pear_mat['fullMatrix'])
    kn = findKneePoint(xs, ys)
    graphThresholdValues(xs, ys, kneePoint=kn, title='Pearson')
    featureGrouping = compressMatrixValues(pear_mat['fullMatrix'], pear_mat['fullList'], 0.1)
    print('Pearson', featureGrouping)
    graphFeatureGroupsSubplots(df, featureGrouping, title='Pearson')

    # spearman
    graphMatrixDistribution(spr_mat['fullMatrix'], title='Spearman')
    graphMatrixHeatmap(spr_mat,title = 'Spearman')
    xs, ys = getThresholdandNumFeaturesLeft(spr_mat['fullMatrix'])
    kn = findKneePoint(xs, ys)
    graphThresholdValues(xs, ys, kneePoint=kn, title='Spearman')
    featureGrouping = compressMatrixValues(spr_mat['fullMatrix'], spr_mat['fullList'], 0.1)
    print('Spearman', featureGrouping)
    graphFeatureGroupsSubplots(df, featureGrouping, title='Spearman')
