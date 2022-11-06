# This file is application of K-means algorithm for MNIST dataset clustering.
# Novementer, 2022, Qingyu Song

# Based on https://www.tutorialspoint.com/scikit_learn/scikit_learn_clustering_methods.htm.
# MNIST loading is based on https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python.
# Plot is based on https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

from mnist import MNIST
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle, islice
from scipy import spatial
from scipy.io import arff

sns.set()

def plot_centers(tag, centers):
    fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    plt.savefig("results\\" + tag)

def plot_scatters(tag, data, label):
    colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(label) + 1),
                )
            )
        )

        # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[label])
    plt.savefig("results\\" + tag)


def cal_centers(data, clusters, dimensions):
    centers = np.zeros(dimensions)
    for i in range(dimensions[0]):
        # print(np.where(clusters==i))
        indices = np.where(clusters==i)[0]
        if len(indices) > 0:
            # np.take(data, indices, axis=0)
            centers[i] = data[indices, ...].mean(axis=0)
    return centers


def metric_accuracy(tag, target, clusters, classes=10):
    from scipy.stats import mode
    from sklearn.metrics import accuracy_score
    labels = np.zeros_like(clusters)
    for i in range(classes):
        mask = (clusters == i)
        labels[mask] = mode(target[mask])[0]
    print(tag, accuracy_score(target, labels))
 

def metric_inter_cluster(tag, metric, centers):
    results = 0
    for i in range(centers.shape[0]):
        for j in range(centers.shape[0]):
            if i != j:
                results += metric(centers[i], centers[j])
    print(tag, results)


def metric_intra_cluster(tag, metric, data, label, classes):
    results = np.zeros(classes)
    for i in range(classes):
        indices = np.where(label==i)[0]
        if len(indices) == 0: continue
        _data = data[indices]
        
        tmp = 0
        shape = _data.shape[0]
        for d1 in range(shape):
            for d2 in range(shape):
                if d2 != d1:
                    tmp += metric(_data[d1], _data[d2])
                    # print(metric(_data[d1, ...], _data[d2, ...]))
        results[i] = tmp / (len(indices)*(len(indices)-1))
    # print(tag, results)
    print(tag, results.sum())



def metric_norm(v1, v2):
    return np.linalg.norm(v1 - v2, ord=2)

def metric_cos(v1, v2):
    return 1 - np.dot(v1,v2)/((np.linalg.norm(v1)*np.linalg.norm(v2)) + 1e-10)
        
def test_mnist():
    mndata = MNIST('datasets')

    data, labels = mndata.load_testing()
    data = np.array(data)
    data = data[:1000]
    labels = np.array(labels)
    labels = labels[:1000]
    size = 28
    num_clusters = 10

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    dbscan = DBSCAN(eps=0.1, min_samples=20, metric='cosine')
    hcluster = AgglomerativeClustering(n_clusters=num_clusters)

    clusters = kmeans.fit_predict(data)
    clusters_dbscan = dbscan.fit_predict(data)
    clusters_hcluster = hcluster.fit_predict(data)

    # print(clusters_dbscan)

    clusters_dbscan += 1

    # print(clusters_dbscan)

    data = data.reshape((-1, size, size))
    kmeans_centers = kmeans.cluster_centers_.reshape(num_clusters, size, size)
    dbscan_centers = cal_centers(data, clusters_dbscan, (num_clusters, size, size))
    hcluster_centers = cal_centers(data, clusters_hcluster, (num_clusters, size, size))

    plot_centers(str(num_clusters) + "clusters_kmeans.png", kmeans_centers)
    plot_centers(str(num_clusters) + "clusters_dbscan.png", dbscan_centers)
    plot_centers(str(num_clusters) + "hcluster.png", hcluster_centers)

    metric_accuracy(str(num_clusters) + "clusters_K-Means", labels, clusters)
    metric_accuracy(str(num_clusters) + "DBSCAN", labels, clusters_dbscan)
    metric_accuracy(str(num_clusters) + "clusters_Agglomerative Clustering", labels, clusters_hcluster)

    label_centers = cal_centers(data, labels, (num_clusters, size, size))
    plot_centers(str(num_clusters) + "clusters_labels.png", label_centers)

    data = data.reshape((-1, size*size))
    kmeans_centers = kmeans.cluster_centers_
    dbscan_centers = cal_centers(data, clusters_dbscan, (num_clusters, size*size))
    hcluster_centers = cal_centers(data, clusters_hcluster, (num_clusters, size*size))

    metric_intra_cluster(str(num_clusters) + "clusters Intra K-Means Euclidean Distance ", metric_norm, data, clusters, num_clusters)
    metric_intra_cluster(str(num_clusters) + "clusters Intra DBSCAN Euclidean Distance ", metric_norm, data, clusters_dbscan, num_clusters)
    metric_intra_cluster(str(num_clusters) + "clusters Intra Agglomerative Clustering Euclidean Distance ", metric_norm, data, clusters_hcluster, num_clusters)

    metric_inter_cluster(str(num_clusters) + "clusters Inter K-Means Euclidean Distance ", metric_norm, kmeans_centers)
    metric_inter_cluster(str(num_clusters) + "clusters Inter DBSCAN Euclidean Distance ", metric_norm, dbscan_centers)
    metric_inter_cluster(str(num_clusters) + "clusters Inter Agglomerative Clustering Euclidean Distance ", metric_norm, hcluster_centers)


    metric_intra_cluster(str(num_clusters) + "clusters Intra K-Means Cosine Similarity", metric_cos, data, clusters, num_clusters)
    metric_intra_cluster(str(num_clusters) + "clusters Intra DBSCAN Cosine Similarity", metric_cos, data, clusters_dbscan, num_clusters)
    metric_intra_cluster(str(num_clusters) + "clusters Intra Agglomerative Clustering Cosine Similarity", metric_cos, data, clusters_hcluster, num_clusters)

    metric_inter_cluster(str(num_clusters) + "clusters Inter K-Means Cosine Similarity", metric_cos, kmeans_centers)
    metric_inter_cluster(str(num_clusters) + "clusters Inter DBSCAN Cosine Similarity", metric_cos, dbscan_centers)
    metric_inter_cluster(str(num_clusters) + "clusters Inter Agglomerative Clustering Cosine Similarity", metric_cos, hcluster_centers)


def test_banana():
    import pandas as pd
    data = arff.loadarff('datasets\\banana.arff')
    df = pd.DataFrame(data[0])
    for i in range(len(df)):
        df["class"][i] = int.from_bytes(df["class"][i], byteorder='big') % 2
    
    data = df.to_numpy()
    labels = data[:, 2]
    data = data[:, :2].astype('float32')
    labels = (labels + 1) % 2
    labels = labels.astype('int32')
    print(data.shape)
    print(labels.shape)

    num_clusters = 4

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    dbscan = DBSCAN(eps=0.05, min_samples=5) # 0.1 0.01
    hcluster = AgglomerativeClustering(n_clusters=num_clusters)

    clusters = kmeans.fit_predict(data)
    clusters_dbscan = dbscan.fit_predict(data)
    clusters_hcluster = hcluster.fit_predict(data)

    # print(clusters_dbscan)
    # print(labels)
    # plot_scatters("banana_" + str(num_clusters) + "clusters_kmeans.png", data, clusters)
    # plot_scatters("banana_" + str(num_clusters) + "clusters_dbscan.png", data, clusters_dbscan)
    # plot_scatters("banana_" + str(num_clusters) + "hcluster.png", data, clusters_hcluster)
    # plot_scatters("banana_" + "labels.png", data, list(labels))
    # print(clusters.shape)
    # print(labels.shape)
    # metric_accuracy("banana_" + str(num_clusters) + "clusters_K-Means", labels, clusters, 2)
    # metric_accuracy("banana_" + str(num_clusters) + "DBSCAN", labels, clusters_dbscan, 2)
    # metric_accuracy("banana_" + str(num_clusters) + "clusters_Agglomerative Clustering", labels, clusters_hcluster, 2)

    # label_centers = cal_centers(data, labels, (num_clusters, size, size))
    # plot_centers(str(num_clusters) + "clusters_labels.png", label_centers)

    # data = data.reshape((-1, size*size))
    kmeans_centers = kmeans.cluster_centers_
    dbscan_centers = cal_centers(data, clusters_dbscan, (num_clusters, 2))
    hcluster_centers = cal_centers(data, clusters_hcluster, (num_clusters, 2))

    # print(kmeans_centers)
    # print(dbscan_centers)
    # print(hcluster_centers)

    # metric_intra_cluster("banana_" + str(num_clusters) + "clusters Intra K-Means Euclidean Distance ", metric_norm, data, clusters, num_clusters)
    # metric_intra_cluster("banana_" + str(num_clusters) + "clusters Intra DBSCAN Euclidean Distance ", metric_norm, data, clusters_dbscan, num_clusters)
    # metric_intra_cluster("banana_" + str(num_clusters) + "clusters Intra Agglomerative Clustering Euclidean Distance ", metric_norm, data, clusters_hcluster, num_clusters)

    # metric_inter_cluster("banana_" + str(num_clusters) + "clusters Inter K-Means Euclidean Distance ", metric_norm, kmeans_centers)
    # metric_inter_cluster("banana_" + str(num_clusters) + "clusters Inter DBSCAN Euclidean Distance ", metric_norm, dbscan_centers)
    # metric_inter_cluster("banana_" + str(num_clusters) + "clusters Inter Agglomerative Clustering Euclidean Distance ", metric_norm, hcluster_centers)


    # metric_intra_cluster("banana_" + str(num_clusters) + "clusters Intra K-Means Cosine Similarity", metric_cos, data, clusters, num_clusters)
    # metric_intra_cluster("banana_" + str(num_clusters) + "clusters Intra DBSCAN Cosine Similarity", metric_cos, data, clusters_dbscan, num_clusters)
    # metric_intra_cluster("banana_" + str(num_clusters) + "clusters Intra Agglomerative Clustering Cosine Similarity", metric_cos, data, clusters_hcluster, num_clusters)

    # metric_inter_cluster("banana_" + str(num_clusters) + "clusters Inter K-Means Cosine Similarity", metric_cos, kmeans_centers)
    # metric_inter_cluster("banana_" + str(num_clusters) + "clusters Inter DBSCAN Cosine Similarity", metric_cos, dbscan_centers)
    # metric_inter_cluster("banana_" + str(num_clusters) + "clusters Inter Agglomerative Clustering Cosine Similarity", metric_cos, hcluster_centers)

test_banana()

# test_mnist()

