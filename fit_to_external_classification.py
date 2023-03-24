import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import mannwhitneyu

import clustering


def plot_external_tag_distribution(data_set_number, points, labels):
    """
    Plot the distribution of the external tags.
    :param data_set_number: the number of the datd set
    :param points: the reduced data
    :return: None
    """
    print('real unique labels', np.unique(labels), len(labels))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='Paired', alpha=0.8, s=8)
    plt.savefig('tag_dist_%i' % data_set_number)
    plt.show()


def nmi_score(labels_true, points, n_clusters, method):
    """
    Returns a list with 20 nmi scores.
    :param labels_true: the real labels
    :param points: the points to cluster
    :param n_clusters: the number of clusters
    :param method: clustering method
    :param linkage: if the method is Hierarchical than linkage represents the sub method
    :returns: a list with 20 nmi scores
    """
    score = []
    if method == 'HDBSCAN':
        for i in range(0, 20):
            labels_pred = clustering.perform_hdbscan(points)
            score.append(normalized_mutual_info_score(labels_true, labels_pred))
    elif method == 'OPTICS':
        for i in range(0, 20):
            labels_pred = clustering.perform_optics(points)
            score.append(normalized_mutual_info_score(labels_true, labels_pred))
    elif method == 'KMeansMiniBatch':
        for i in range(0, 20):
            labels_pred = clustering.perform_minibatch(points, n_clusters)
            score.append(normalized_mutual_info_score(labels_true, labels_pred))
    elif method == 'Hierarchical':
        for i in range(0, 20):
            labels_pred = clustering.perform_hierarchical_clustering(points, n_clusters)
            score.append(normalized_mutual_info_score(labels_true, labels_pred))
    elif method == 'Birch':
        for i in range(0, 20):
            labels_pred = clustering.perform_birch(points, n_clusters)
            score.append(normalized_mutual_info_score(labels_true, labels_pred))
    else:
        for i in range(0, 20):
            labels_pred = clustering.perform_kmeans(points, n_clusters)
            score.append(normalized_mutual_info_score(labels_true, labels_pred))
    return score


def u_test(scores_method_1, scores_method2):
    """
    Returns P value. if p<<0.05 the first scores better than the second
    :param scores_method_1: first method's scores
    :param scores_method2: second method's scores
    :returns: p value
    """
    mann_whitneyu = mannwhitneyu(scores_method_1, scores_method2, alternative='greater')
    # if p value<0.05 than we can say nmi1>nmi2. Therefore, clustering method 1 is better than 2.
    return mann_whitneyu.pvalue
