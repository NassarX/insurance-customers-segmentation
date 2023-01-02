import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from sklearn.cluster import KMeans
import os
import warnings
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")


def load_data(file_path):
    """Loads data from a file and returns a Pandas dataframe.

    :param: path: str The file path of the data files.

    :returns pandas.DataFrame The data as a Pandas dataframe.
    """
    # load train data sets
    df = pd.read_sas(file_path)

    return df


def detect_outliers(data, metrics, method='iqr'):
    """ Detects outliers of data using the IQR,  Z-score or empirical rule method.

    :param:
    data: pandas.DataFrame - The data to be analyzed.
    method: string, optional - emp, z_score, iqr

    :returns
    pandas.DataFrame
    """

    outliers_summary = {}
    outliers = pd.DataFrame()
    if method == 'iqr':
        """Identify the  outliers by IQR rule."""

        q25 = data.quantile(.25)
        q75 = data.quantile(.75)
        iqr = (q75 - q25)
        upper_lim = q75 + 1.5 * iqr
        lower_lim = q25 - 1.5 * iqr

        for metric in metrics:
            llim = lower_lim[metric]
            ulim = upper_lim[metric]
            outlier_rows = data[(data[metric] < llim) | (data[metric] > ulim)]
            if len(outlier_rows) > 0:
                outliers = outliers.append(outlier_rows)
                outliers_summary[metric] = {'lower': round(llim, 2), 'upper': round(ulim, 2),
                                            'count': len(outlier_rows)}

        return [outliers_summary, outliers.drop_duplicates().sort_index()]
    elif method == 'z_score':
        for metric in metrics:

            # Calculate the mean and standard deviation
            data_mean, data_std = np.mean(data[metric]), np.std(data[metric])

            # Calculate the z-scores for each value
            z_scores = zscore(data[metric])

            # Set the threshold for identifying outliers
            threshold = 3

            outlier_rows = data[(abs(z_scores) > threshold)]
            if len(outlier_rows) > 0:
                outliers = outliers.append(outlier_rows)
                outliers_summary[metric] = {'threshold': threshold, 'count': len(outlier_rows)}
        return [outliers_summary, outliers.drop_duplicates().sort_index()]


def get_education_rank(degree):
    """ Define function for ranking education degree

    :param:
    Education Degree
    :return: string
    """

    if degree == b'1 - Basic':
        return 1
    elif degree == b'2 - High School':
        return 2
    elif degree == b'3 - BSc/MSc':
        return 3
    elif degree == b'4 - PhD':
        return 4
    else:
        return degree


def heatmap_corr(cor, figures_path, title):
    p_corr = round(cor.iloc[1:, :-1].copy(), 2)

    # Setting up a diverging palette
    #    plt.subplots(figsize=(12, 6))

    # Prepare figure
    fig = plt.figure(figsize=(10, 8))

    # Build annotation matrix (values above |0.5| will appear annotated in the plot)
    mask_annot = np.absolute(p_corr.values) >= 0.5
    annot = np.where(mask_annot, p_corr.values,
                     np.full(p_corr.shape, ""))  # Try to understand what this np.where() does

    # Plot heatmap of the correlation matrix
    sns.heatmap(data=p_corr, annot=annot, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                fmt='s', vmin=-1, vmax=1, center=0, square=True, linewidths=.5)

    # Layout
    fig.subplots_adjust(top=0.95)

    plt.title(title)
    plt.savefig(os.path.join(figures_path, title.replace(" ", "_") + '_heatmap.png'),
                dpi=200)
    plt.show()


def elbow_method(data):
    # find 'k' value by Elbow Method
    inertia = []
    range_val = range(1, 10)
    for i in range_val:
        kmean = KMeans(n_clusters=i)
        kmean.fit_predict(pd.DataFrame(data))
        inertia.append(kmean.inertia_)

    plt.plot(range_val, inertia, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()


def hyperparameter_tuning(model_, grid_params, data, scoring='f1', cv=5):
    """adjusting the hyperparameters of a model to improve its performance."""
    grid = GridSearchCV(estimator=model_, param_grid=grid_params, scoring=scoring, cv=cv)

    grid.fit(data)
    model_grid_best = grid.best_estimator_
    best_params_ = grid.best_params_

    return grid, best_params_


def silhouette_method(data):
    # Storing average silhouette metric
    avg_silhouette = []

    range_val = range(2, 10)
    for n_clus in range_val:
        # Initialize the KMeans object with n_clusters value and a random generator
        kmeans_cluster = KMeans(n_clusters=n_clus, init='k-means++', n_init=15, random_state=1)
        cluster_labels = kmeans_cluster.fit_predict(data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        avg_silhouette.append(silhouette_avg)
        print(f"For n_clusters = {n_clus}, the average silhouette_score is : {silhouette_avg}")

    # Find the index of the highest silhouette score
    optimal_k = np.argmax(avg_silhouette) + 2
    # Print the optimal value of k
    print("Optimal number of clusters:", optimal_k)

    # The average silhouette plot
    # The inertia plot
    plt.plot(avg_silhouette)
    plt.ylabel("Average silhouette")
    plt.xlabel("Number of clusters")
    plt.title("Average silhouette plot over clusters", size=15)
    plt.show()

    return optimal_k


def plot_dendrogram(data, linkage_='ward', distance='euclidean'):
    """Plot corresponding dendrogram"""

    linkage_matrix = linkage(data, linkage_)

    sns.set()
    fig = plt.figure(figsize=(11, 5))
    # The Dendrogram parameters need to be tuned
    y_threshold = 100
    dendrogram(linkage_matrix, truncate_mode='level', p=5, color_threshold=y_threshold, above_threshold_color='k')
    plt.hlines(y_threshold, 0, 1000, colors="r", linestyles="dashed")
    plt.title(f'Hierarchical Clustering - {linkage_.title()}\'s Dendrogram', fontsize=21)
    plt.xlabel('Number of points in node (or index of point if no parenthesis)')
    plt.ylabel(f'{distance.title()} Distance', fontsize=13)

    plt.savefig(os.path.join('assets/figures', 'kmeans_centroids_parallel_plot.png'), dpi=200)
    plt.show()


def get_ss(df):
    """Computes the sum of squares for all variables given a dataset

    :param df dataset
    :returns return sum of squares of each df variable
    """

    ss = np.sum(df.var() * (df.count() - 1))
    return ss


def r2_score(df, labels):
    """Get the R2 metrics for each cluster solution

    :param df datasets
    :param labels cluster labels

    :returns R2 metrics for each cluster
    """

    # get total sum of squares
    sst = get_ss(df)

    # compute ssw for each cluster labels
    ssw = np.sum(df.groupby(labels).apply(get_ss))

    return 1 - ssw / sst


def models_scores(df, models=None):
    if models is None:
        models = {}

    r2_scores = {}
    silhouette_scores = {}
    kmeans_scores = {'r2_score': {}, 'silhouette_score': {}}
    agglo_scores = {'r2_score': {}, 'silhouette_score': {}}
    range_val = range(2, 10)
    for n_clus in range_val:
        agglo_model_ = models['agglo'].set_params(n_clusters=n_clus)
        agglo_labels_ = agglo_model_.fit_predict(df)
        agglo_scores['r2_score'][n_clus] = r2_score(df, agglo_labels_)
        agglo_scores['silhouette_score'][n_clus] = silhouette_score(df, agglo_labels_)

        kmeans_ = models['kmeans'].set_params(n_clusters=n_clus)
        kmeans_labels_ = kmeans_.fit_predict(df)
        kmeans_scores['r2_score'][n_clus] = r2_score(df, kmeans_labels_)
        kmeans_scores['silhouette_score'][n_clus] = silhouette_score(df, kmeans_labels_)

    r2_scores['K-means'] = kmeans_scores['r2_score']
    r2_scores['Agglomerative'] = agglo_scores['r2_score']

    silhouette_scores['K-means'] = kmeans_scores['silhouette_score']
    silhouette_scores['Agglomerative'] = agglo_scores['silhouette_score']

    return pd.DataFrame(r2_scores), pd.DataFrame(silhouette_scores)


class Helper:
    def __init__(self):
        pass
