import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, Birch, OPTICS
import matplotlib.pyplot as plt
import skfuzzy


def perform_fuzzy_cmeans(points, n_clusters):
    """
    Perform FCM clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    cntr, u, _, _, _, _, _ = skfuzzy.cluster.cmeans(points.T, c=n_clusters, m=2, error=0.005, maxiter=1000)
    predictions = np.argmax(u, axis=0)
    return predictions


def perform_hdbscan(points):
    """
    Perform HDBSCAN and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    hdbscan = HDBSCAN(min_cluster_size=15)
    hdbscan.fit(points)

    return hdbscan.labels_


def perform_hierarchical_clustering(points, n_clusters, linkage='ward'):
    """
    Perform Hierarchical clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :param linkage: the sub method
    :returns: clustering labels
    """
    # linkages = ['ward', 'average', 'complete', 'single']
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    predictions = hc.fit_predict(points)
    return predictions


def perform_kmeans(points, n_clusters):
    """
    Perform K means clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    predictions = KMeans(n_clusters=n_clusters).fit_predict(points)
    return predictions


def perform_birch(points, n_clusters):
    """
    Perform Birch clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    predictions = Birch(n_clusters=n_clusters).fit_predict(points)
    return predictions


def perform_minibatch(points, n_clusters):
    """
    Perform KMeansMiniBatch clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    predictions = MiniBatchKMeans(n_clusters=n_clusters).fit_predict(points)
    return predictions



def perform_optics(points):
    """
    Perform OPTICS clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    predictions = OPTICS().fit_predict(points)
    return predictions



def plot_clustering(points, predictions, method):
    """
    Visualize the clustering results
    :param points: points to plot
    :param predictions: points labels according to cluster algorithms
    :param method: clustering method
    :return: None
    """
    path = method + ".png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    ax.scatter(points[:, 0], points[:, 1], c=predictions, cmap='tab10', alpha=0.8, s=8)
    plt.savefig(path)
    plt.show()
