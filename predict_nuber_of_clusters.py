import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

import clustering
import fit_to_external_classification


def compare_external_variables(points, method, n_clusters_list):
    """
    Compare silhouette scores with u test to find which external variable best expalains the data.
    :param points: the data reduced points
    :param method: clustering method
    :param n_clusters_list: the number of labels fo reach external variable
    :return: None
    """
    n_clusters_silhouette_dictionary = {}
    for n_clusters in n_clusters_list:
        n_clusters_silhouette_dictionary[n_clusters] = get_20_times_silhouette_score(points, method, n_clusters)
    for key1 in n_clusters_silhouette_dictionary:
        for key2 in n_clusters_silhouette_dictionary:
            if key1 != key2:
                pvalue = fit_to_external_classification.u_test(n_clusters_silhouette_dictionary[key1],
                                                               n_clusters_silhouette_dictionary[key2])
                if pvalue < 0.05:
                    print('for', method, ' ', key1, 'clusters are better than', key2,
                          'clusters, with p value =', pvalue, ' <<0.05')



def compare_silhouette_scores(points, method, linkage=''):
    """
    Compare silhouette scores with u test to find best parameters for the method.
    :param points: the data reduced points
    :param method: clustering method
    :param linkage: sub method - if hierarchical clustering
    :return: None
    """
    n_clusters_silhouette_dictionary = {}
    for n_clusters in range(2, 8):
        n_clusters_silhouette_dictionary[n_clusters] = get_20_times_silhouette_score(points, method, n_clusters,
                                                                                     linkage)
    for key1 in n_clusters_silhouette_dictionary:
        for key2 in n_clusters_silhouette_dictionary:
            if key1 != key2:
                pvalue = fit_to_external_classification.u_test(n_clusters_silhouette_dictionary[key1],
                                                               n_clusters_silhouette_dictionary[key2])
                if pvalue < 0.05:
                    print('for', method, ' ', linkage, ': ', key1, 'clusters are better than', key2,
                          'clusters, with p value =', pvalue, ' <<0.05')



def get_20_times_silhouette_score(points, method, n_clusters=0, linkage='ward', labels=None):
    """
    Return a list of 20 times the silhouette scores
    :param points: data reduced points
    :param method: clustering methods
    :param n_clusters: number of clusters
    :param linkage: sub method - if hierarchical
    :return: a list with 20 times the silhouette score
    """
    scores = []
    if method == 'HDBSCAN':
        for i in range(0, 20):
            labels_pred = clustering.perform_hdbscan(points)
            scores.append(get_silhouette_score(points, labels_pred))
    elif method == 'OPTICS':
        for i in range(0, 20):
            labels_pred = clustering.perform_optics(points)
            scores.append(get_silhouette_score(points, labels_pred))
    elif method == 'KMeansMiniBatch':
        for i in range(0, 20):
            labels_pred = clustering.perform_minibatch(points, n_clusters)
            scores.append(get_silhouette_score(points, labels_pred))
    elif method == 'Hierarchical':
        for i in range(0, 20):
            labels_pred = clustering.perform_hierarchical_clustering(points, n_clusters, linkage)
            scores.append(get_silhouette_score(points, labels_pred))
    elif method == 'Birch':
        for i in range(0, 20):
            labels_pred = clustering.perform_birch(points, n_clusters)
            scores.append(get_silhouette_score(points, labels_pred))
    else:
        for i in range(0, 20):
            labels_pred = clustering.perform_kmeans(points, n_clusters)
            scores.append(get_silhouette_score(points, labels_pred))
    return scores



def get_silhouette_score(points, labels):
    """
    Return the silhouette score
    :param points: data reduced points
    :param labels: data labels according to clustering method
    :return: silhouette score
    """
    return silhouette_score(points, labels)


def perform_elbow_method(points, method, linkage='ward'):
    """
    Perform and visualize elbow method.
    :param points: the data's points
    :param method: clustering method - K means or Hierarchical
    :return: None
    """
    if method == 'K means':
        model = KMeans()
        path = "Kelbow.png"
    elif method == 'Hierarchical':
        model = AgglomerativeClustering(linkage=linkage)
        path = "Helbow.png"
    else:
        raise Exception('This elbow method designed only for K means and Hierarchical')
    visualizer = KElbowVisualizer(model, k=(1, 12), title=" ")
    # Fit the data to the visualizer
    visualizer.fit(points)
    visualizer.show(outpath=path)


def perform_silhouette_method(points, method):
    """
    Calculate and visualize silhouette scores
    :param points: data's points
    :param method: clustering method
    :return: None
    """    
    visualgrid = []
    
    n_clusters = [2, 3, 4, 5, 6, 7]
            
    for i in range(6): 
        path = method + "_" + str(n_clusters[i]) + ".png" 
        
        fig = plt.figure()
        fig.set_size_inches(18, 7)
        ax = fig.add_subplot(111)

         # Instantiate the clustering model and visualizer
        if method == 'K means':
            model = KMeans(n_clusters=n_clusters[i])
             # Instantiate the clustering model and visualizer
        elif method == 'Hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters[i])
        else:
            model = MiniBatchKMeans(n_clusters=n_clusters[i])
            
        viz = SilhouetteVisualizer(model, colors='yellowbrick', title=" ")
        viz.fit(points)
        viz.finalize()
        print("For n_clusters =", n_clusters[i], "The average silhouette_score is :", viz.silhouette_score_)
        
        plt.savefig(path)
    plt.show()
    

    
"""
def perform_silhouette_method(points, method):
    
    Calculate and visualize silhouette scores
    :param points: data's points
    :param method: clustering method
    :return: None
    
    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        path = method + "_" + str(n_clusters) + ".png"
        # Create a figure
        fig = plt.figure()
        fig.set_size_inches(18, 7)
        ax = fig.add_subplot(111)
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(points) + (n_clusters + 1) * 10])

        # find the labels for the clustering method and number of clusters
        cluster_labels = clustering.cluster(points, n_clusters, method)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(points, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(points, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        label = "The Silhouette score\nis " + str(silhouette_avg)[:8]
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=label)

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        ax.legend(loc ="upper right")
        
        plt.savefig(path)
    plt.show()
"""