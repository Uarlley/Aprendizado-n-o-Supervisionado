from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
from yellowbrick.cluster import silhouette_visualizer

plt.style.use('default')

plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 2,
    'axes.titlesize': 20,
    'axes.edgecolor': 'black',
    'axes.labelsize': 20,
    'axes.grid': True,
    'lines.linewidth': 1.5,
    'lines.markersize': 3,
    'figure.figsize': (15, 8),
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'font.family': 'Arial',
    'legend.fontsize': 13,
    'legend.framealpha': 1,
    'legend.edgecolor': 'black',
    'legend.shadow': False,
    'legend.fancybox': True,
    'legend.frameon': True,
})

def max_index(array):
    """Gets the index of the largets item of an given array.

    Args:
        array (list, numpy array): an list or numpy array

    Returns:
        integer: index of the largest item
    """
    aux = np.array(array)
    return np.where(aux == np.amax(aux))[0][0]


def min_index(array):
    """Gets the index of the smallest item of an given array.

    Args:
        array (list, numpy array): an list or numpy array

    Returns:
        integer: index of the smallest item
    """
    aux = np.array(array)
    return np.where(aux == np.amin(aux))[0][0]


def normalize_(array):
    """normalizes all values of an given array

    Args:
        array (list, numpy array): array to be normalized

    Returns:
        numpy array: normalized array
    """
    aux = np.array(array)
    return (aux - aux.min()) / (aux.max() - aux.min())


def make_categorical(array):
    new_arr = [0]*len(array)
    for i in range(len(array)):
        new_arr[i] = 'outlier' if array[i] == -1 else 'cluster ' + str(array[i])

    return new_arr


def my_range(begin, end, step):
    array = []
    while (round(begin, 3) < end):
        array += [begin]
        begin += step
    return array

def make_categorical(array, label = 'Cluster'):
    new_array = []
    for i in array:
        new_array += [str(i)]
    return new_array

def check_nan(dataframe, plot = True):
    
    """Checks whether a given dataset contains any missing value or not

    Args:
        dataframe (pandas): Input dataset
        plot (boolean, optional): If the heatmap should be displayed. Defaults to True.
        
    Returns:
        (boolean): if the dataframe contains missing values
    """
    if(plot):
        ax = sns.heatmap(dataframe.isna())
        plt.show()

    nans = dataframe.isna()
    columns = nans.columns

    for i in range(len(columns)):
        if True in list(nans[columns[i]]):
            return True

    return False


def remove_nan(dataframe, axis = 0, plot = True):

    """Removes all missing values from a given pandas dataframe.

    Args:
        dataframe (pandas): Input dataset
        axis (0 or 'index', 1 or 'columns', optional): Determine if rows or columns 
        which contain missing values are removed. Defaults to 0.
             - 0, or 'index' : Drop rows which contain missing values.
             - 1, or 'columns' : Drop columns which contain missing value.
        plot (boolean, optional): If the heatmap should be displayed. Defaults to True.
        
    Returns:
        (pandas): DataFrame with NA entries dropped from it.
    """
    new_df = dataframe.copy()

    if(check_nan(dataframe)):
        new_df  = new_df.dropna(axis = axis)

    if(plot):
        ax = sns.heatmap(new_df.isna())
        plt.show()
    return new_df


class kmeans():
    
    def __init__(self, df, cat=False):
        if(type(df) != pd.DataFrame):
            print("data must be a pandas DataFrame!")
        else:
            self.X = df.copy()
            self.model = None
            self.labels = None

    def clustering(self, X):
        """Predict the labels for the data samples in X using the trained model.
        Args:
            X (nd array(n_samples, n_features)): List of n_features-dimensional data points. Each row corresponds to a single data point.

        Returns:
            labels (list): Component labels.
        """
        if not self.model:
            print("Create a clustering model first")
            return

        self.labels = self.model.predict(X)
        return self.labels

    def get_data(self):
        """Get method for the class dataset

        Returns:
            array like (n_samples, n_features): returns the class dataset
        """
        return self.X

    def get_model(self):
        """Get method for the class model

        Returns:
            clustering model: returns the child respective clustering model
        """
        if not self.model:
            print("Create a clustering model first")
        return self.model

    def silhouette_score_(self):
        try:
            return silhouette_score(self.X, self.labels)
        except ValueError:
            print("The number of clusters must be greater than 1!")
            return

    def calinski_harabasz_score_(self):
        try:
            return calinski_harabasz_score(self.X, self.labels)
        except ValueError:
            print("The number of clusters must be greater than 1!")
            return

    def davies_bouldin_score_(self):
        try:
            return davies_bouldin_score(self.X, self.labels)
        except ValueError:
            print("The number of clusters must be greater than 1!")
            return
        
    def cluster_evaluation(self, n_clusters, verbose=False, normalize=True):

        if (n_clusters <= 2):
            print("n_clusters must be greater than 2.")
            return

        ine = []
        sil = []
        calinski_harabasz = []
        davies_bouldin = []

        n_range = range(2, n_clusters + 1)

        for i in tqdm(n_range):

            # creating, fitting and predicting the model
            km = kmeans(self.X)
            km.create_model(n_clusters=i)
            km.clustering(self.X)

            # computing the scores
            ine.append(km.inertia())  # Find Elbow
            sil.append(km.silhouette_score_())  # Maximize
            calinski_harabasz.append(km.calinski_harabasz_score_())  # Maximize
            davies_bouldin.append(km.davies_bouldin_score_())  # Minimize

            if verbose:
                print(f'Number of groups: {i} --- Inertia: {ine[i - 2]}')
                print(f'Number of groups: {i} --- Silhouette coefficient: {sil[i - 2]}')
                print(f'Number of groups: {i} --- Calinski_harabasz: {calinski_harabasz[i - 2]}')
                print(f'Number of groups: {i} --- Davies_bouldin: {davies_bouldin[i - 2]}\n')

        # normalizing the scores
        if normalize:
            ine = normalize_(ine)
            sil = normalize_(sil)
            calinski_harabasz = normalize_(calinski_harabasz)
            davies_bouldin = normalize_(davies_bouldin)

        # Index of optimal metric value
        kl = KneeLocator(n_range, ine, curve="convex", direction='decreasing', S=2)

        ylabels = ["Inertia", "Silhouette coefficient", "Calinski harabasz score", "Davies bouldin score"]
        scores = [ine, sil, calinski_harabasz, davies_bouldin]
        indexes = [kl.knee, max_index(sil), max_index(calinski_harabasz), max_index(davies_bouldin)]

        # Plotting
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(26, 13))
        plt.setp(ax[1], xlabel='Number of groups')

        for i in range(4):
            ax[i // 2][i % 2].plot(n_range, scores[i])
            ax[i // 2][i % 2].set_ylabel(ylabels[i])
            if (indexes[i] != None):
                ax[i // 2][i % 2].axvline(x=n_range[indexes[i]], ymax=scores[i].max(), color='red', linestyle='--')
            else:
                print("No knee/elbow found")

        [ax[i // 2][i % 2].set_xticks(n_range) for i in range(0, 2)]
        plt.tight_layout()
        plt.show()

    def create_model(self, n_clusters):
        self.model = KMeans(n_clusters=n_clusters, max_iter=5000, random_state=42)
        self.model.fit(self.X)
        return

    def silhouette_plot(self):
        if not self.model:
            print("Create a clustering model first")

        else:
            silhouette_visualizer(self.model, self.X)

        return

    def inertia(self):

        if not self.model:
            print("Create a clustering model first.")
        else:
            return self.model.inertia_



