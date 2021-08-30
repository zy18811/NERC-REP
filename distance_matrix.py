import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool, cpu_count
from dtaidistance import dtw
from tqdm import tqdm
from july21_get_all_data import get_july21_all_data
from sklearn import preprocessing
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import ks_2samp

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
    res = spearmanr(df[col1].to_numpy(),df[col2].to_numpy())
    mod_res = 1-abs(res[0])

    return mod_res


def pearson(df,col1,col2):
    res = pearsonr(df[col1].to_numpy(), df[col2].to_numpy())
    mod_res = 1-abs(res[0])

    return mod_res

def ks_test(df,col1,col2):
    res = ks_2samp(df[col1].to_numpy(), df[col2].to_numpy())
    return res[0]



def doDTW(index, currentCol, df, distFunc):
    #print(f"Column {index} started")
    currentRow = []

    cols=df.columns[:index] # Columns we need to compare too, changes each index to not repeat combinations
    for col in cols:
        currentRow.append(distFunc(df,currentCol,col))
    currentRow.append(0) # After len(cols), will be the comparison to itself, i.e. 0 distance

    for n in range(len(df.columns)-1-index):
        currentRow.append(np.nan) # Remaining values in row can be nans as will be repeats


    #print(f"Column {index} finished")
    return currentRow


def doDTWWrapper(args):
    return doDTW(*args)


def distMatrix(df,distFunc):
    results = dict()
    results["fullList"] = df.columns

    args = [(i, col, df) for i, col in enumerate(df.columns)]


    # Do each column seperately
    dtwRowList = []
    for a in tqdm(args):
        dtwRowList.append(doDTW(*a,distFunc))

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
    groupedFeaturesList = [(colNames[featureIndex1],colNames[featureIndex2]) for featureIndex1, featureIndex2 in indexes]
    # List of pairs of sensor names that have distance less than the threshold


    return merge(groupedFeaturesList)


def graphMatrixDistribution(matrix,title = None):

    values = matrix.tolist()
    values = np.delete(values, np.where(values==0))
    values = np.delete(values, np.where(np.isnan(values)))
    plt.figure()
    if title!=None:
        plt.title(title)
    plt.hist(values,bins = 'auto',ec = 'black')
    #plt.ylim(10)
    plt.xlabel("Distance Value")
    plt.ylabel("Frequency")
    plt.show()



def getThresholdandNumFeaturesLeft(matrix):
    """
    For some matrix of similarity values, calculates the number of groups of features for different threshold values
    :param matrix:
    :return:
    """
    colNames = [n for n in range(len(matrix))]

    uniqueVals = np.delete(np.unique(matrix),np.where(np.isnan(np.unique(matrix))))

    thresholdIndexes = np.unique(np.linspace(zeroValue,len(uniqueVals)-1,1000).astype(int))
    #Caps it to up to 1000 thresholding values, equally spaced indexes


    thresholdValues = np.array([uniqueVals[index] for index in thresholdIndexes])

    numOfGroups = []

    for threshold in tqdm(thresholdValues,desc="Threshold Num Groups"):
        numOfGroups.append(len(compressMatrixValues(matrix=matrix,colNames=colNames,threshold=threshold)))

    return thresholdValues,numOfGroups

def findKneePoint(xs,ys):
    kn = KneeLocator(xs, ys, curve='convex', direction='decreasing',online=True,interp_method='polynomial',
                     polynomial_degree=7)
    return kn.knee

def graphThresholdValues(xs,ys,kneePoint,title = None):
    plt.figure()
    plt.vlines(kneePoint,min(ys),max(ys), linestyles='dashed',label = 'Raw Knee')


    plt.plot(xs, ys, label='Raw')

    if title!=None:
        plt.title(title)
    plt.xlabel(f"Threshold values, raw knee at {kneePoint:.2f}")
    plt.ylabel(f"Number of grouped features, raw knee at {(ys[list(xs).index(kneePoint)]):.2f}")
    plt.legend()
    plt.show()


def graphFeatureGroupsSubplots(df,featureGrouping,title = None):
    plt.figure()
    if title!=None:
        plt.suptitle(title)
    count = 1
    for group in featureGrouping:
        plt.subplot(len(featureGrouping), 1, count)
        for col in group:
            plt.plot(df[col].index.values, df[col].values, label=col)
        plt.legend()
        count += 1
    plt.show()




if __name__ == '__main__':
    data_df = get_july21_all_data()
    df = data_df

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)

    dtw_mat = distMatrix(df,calcDTWDist)
    pear_mat = distMatrix(df,pearson)
    spr_mat = distMatrix(df,spearman)
    
    
    #print(mat['fullMatrix'])
    graphMatrixDistribution(dtw_mat['fullMatrix'],title='DTW')
    xs, ys = getThresholdandNumFeaturesLeft(dtw_mat['fullMatrix'])
    kn = findKneePoint(xs, ys)

    graphThresholdValues(xs, ys, kneePoint=kn,title='DTW')
    featureGrouping = compressMatrixValues(dtw_mat['fullMatrix'], dtw_mat['fullList'], 3.25)
    print('DTW',featureGrouping)
    graphFeatureGroupsSubplots(df,featureGrouping,title = 'DTW')


    graphMatrixDistribution(pear_mat['fullMatrix'], title='Pearson')
    xs, ys = getThresholdandNumFeaturesLeft(pear_mat['fullMatrix'])
    kn = findKneePoint(xs, ys)

    graphThresholdValues(xs, ys, kneePoint=kn, title='Pearson')
    featureGrouping = compressMatrixValues(pear_mat['fullMatrix'], pear_mat['fullList'], 0.265)
    print('Pearson', featureGrouping)
    graphFeatureGroupsSubplots(df, featureGrouping, title='Pearson')

    graphMatrixDistribution(spr_mat['fullMatrix'], title='Spearman')
    xs, ys = getThresholdandNumFeaturesLeft(spr_mat['fullMatrix'])
    kn = findKneePoint(xs, ys)


    graphThresholdValues(xs, ys, kneePoint=kn, title='Spearman')
    featureGrouping = compressMatrixValues(spr_mat['fullMatrix'], spr_mat['fullList'], 0.5)
    print('Spearman', featureGrouping)
    graphFeatureGroupsSubplots(df, featureGrouping, title='Spearman')