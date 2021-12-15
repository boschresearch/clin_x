"""
ClusterDataSplit: companion code for the benchmarking study reported in the paper:

    ClusterDataSplit: Exploring Challenging Clustering-Based Data Splits for Model Performance Evaluation by Hanna Wecker, Annemarie Friedrich and Heike Adel. In Proceedings of Evaluation and Comparison of NLP Systems (Eval4NLP).

Copyright (c) 2019 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# load packages
import copy
import random
import collections

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import pandas as pd

import sklearn
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.cluster.k_means_ import _init_centroids
print(sklearn.__version__)
version_nr = int(sklearn.__version__.split(".")[1])
if version_nr > 23:
    print("Careful - ClusterDataSplit only works with sklearn <= version 0.23 !!")



def match_ids_class_labels(gold_labels):
    """
    Assigns an ID for each example and combines it with the label information.
    :param gold_labels: 2-dim array. Label for each input example.
    :return: pandas DataFrame with columns 'ids' and 'class_labels'.
    """
    ids = np.array([range(0, len(gold_labels[0]))])
    ids_labels = np.array([np.squeeze(ids), np.squeeze(gold_labels)])
    id_and_class_labels = pd.DataFrame(ids_labels.T, columns=['ids', 'class_labels'])

    return id_and_class_labels


def set_up_calculation_matrices(vectors, id_and_class_labels):
    """
    initializes a calculation matrix for each label existing in the data.
    :param vectors: 2-dim array. Contains vector representation for each example.
    :param id_and_class_labels: pandas DataFrame with columns 'ids' and 'class_labels'.
    :return: calculation_matrices. List. contains a calculation matrix (2-dim numpy array) for each label in the data.
                Each calculation matrix is filled with initial information (example ID, vector representation).
    """

    # extract how many labels exist in data set
    unique_labels = np.unique(id_and_class_labels['class_labels'])

    calculation_matrices = [None] * len(unique_labels)

    for i in unique_labels:

        # select all ids belonging to current gold label and extract coordinates (vectors)
        ids_gold_label = np.where(id_and_class_labels['class_labels'] == i)[0]
        label_vectors = vectors[ids_gold_label]

        # calculation matrix: ID, 0, 0, 0, coordinates
        calculation_matrix = label_vectors
        id_array = np.array([ids_gold_label]).transpose()
        zero_array = np.array([np.array(np.repeat(0, len(calculation_matrix)))]).transpose()
        calculation_matrix = np.concatenate((id_array, zero_array, zero_array, zero_array, label_vectors), axis=1)

        # TODO: do we want to generalize for non-numeric labels?
        calculation_matrices[int(i)] = calculation_matrix

    return calculation_matrices


def calculate_order_measure(X, centroids, cluster_labels):
    """
    calculates distances from the example to all centroids. Returns the closest cluster and its respective order
    measure. The order measure is defined as the distance from the closest cluster to the farthest cluster.
    :param X: 2-dim array. Contains vector representation for each example.
    :param centroids: 2-dim array. Vector representation of centroids.
    :param cluster_labels: 1-dim np.array. Clusters available.
    :return: cluster_and_order: 2-dim array. Contains closest cluster and distance from nearest cluster to farthest
                cluster (order measure).
    """

    # obtain distance array containing distances from each point to each cluster center
    # returns n_X times n_centroids array with distances
    # in array 1, 1st entry describes distance from first point to first cluster center, 2nd entry describes
    # distance from 1st point to 2nd cluster center etc.
    distance_array = cdist(X, centroids, metric="sqeuclidean")

    # order points by distance to the nearest cluster minus distance to the farthest cluster
    distance_nearest_cluster = np.amin(distance_array, axis=1)
    distance_farthest_cluster = np.amax(distance_array, axis=1)
    order_measure = distance_nearest_cluster - distance_farthest_cluster

    # extract, which cluster is closest to a certain point
    closest_cluster = cluster_labels[np.argmin(distance_array, axis=1)]

    cluster_and_order = np.concatenate((np.array([closest_cluster]).transpose(), np.array([order_measure]).transpose()),
                                       axis=1)

    return cluster_and_order


def sort_calculation_matrix(calculation_matrix):
    """ returns the calculation matrix sorted according to the order measure"""
    return calculation_matrix[calculation_matrix[:, 2].argsort()]


def find_stop(calculation_matrix, final_matrix, centroids_label, max_ids, label_counts, rounds, cluster_labels):
    """
    orders points to clusters until one of the clusters reaches its maximum size. attention: does not respect labels,
    selection of labels has to be conducted before using this method - this is why only one calculation matrix is
    input to the method at a time.
    :param calculation_matrix: 2-dim numpy array. It contains the following information for each
                example that was not assigned to a cluster already: ID, current_cluster, order measure (distance to
                current cluster assigned - distance to farthest cluster), desired_cluster, coordinates.
    :param final_matrix: 2-dim array. Contains same information as calculation matrix, but for examples that were
                 assigned already.
    :param centroids_label: 2-dim array. Vector representation of centroids. (there are no specific centroids for each
                    label. Centroids are only called differently here, because centroids are removed from the
                    list of centroids once the cluster is full for the respective label and the original list
                    of centroids should not be overwritten.
    :param max_ids: 1-dim numpy array. Contains number of examples to be assigned to the different clusters for one
                    specific label.
    :param label_counts: 1-dim array, same dimensions as max_ids. Contains the number of examples aready assigned to
                    the different clusters.
    :param rounds: integer. current round in the execution of the algorithm.
    :param cluster_labels: 1-dim numpy array. Contains cluster labels still available for assignment.
    :return: Returns the same variables as are the inputs. Variable types for each variable are also the same.
                the difference is that in each time this algorithm is called, points are assigned to clusters, thus
                they are not part of the calculation_matrix anymore, but instead become part of the final matrix. The
                cluster that reached its maximum size is removed from the cluster_labels vector. The associated counts
                and vectors are also removed from the variables centroids_label, max_ids, and label_counts.
    """

    # iterates through calculation matrix
    length = len(calculation_matrix)
    for i in range(0, length):
        current_cluster = calculation_matrix[i, 1]
        id = np.where(cluster_labels == current_cluster)
        label_counts[id] = label_counts[id] + 1
        # if one of the clusters reaches its maximum size for the respective label, split calculation matrix here
        # and store points that were assigned up to this point in the final matrix
        if any(label_counts == max_ids):
            if rounds == 0:
                final_matrix = calculation_matrix[:(i+1)]
            else:
                final_matrix = np.concatenate((final_matrix, calculation_matrix[:(i+1)]), axis=0)
            # cut off calculation matrix part that was assigned to cluster
            calculation_matrix = calculation_matrix[(i+1):]
            remove_vector = np.invert(label_counts == max_ids)
            # remove centroid, max_ids and label_counts of cluster that is now full
            centroids_label = centroids_label[remove_vector]
            max_ids = max_ids[remove_vector]
            label_counts = label_counts[remove_vector]
            cluster_labels = cluster_labels[remove_vector]
            rounds += 1
            break

    return calculation_matrix, final_matrix, centroids_label, max_ids, label_counts, rounds, cluster_labels


def first_assignment(X, n_clusters, max_ids_labels, centroids, labels):
    """
    sets up calculation matrices for each label and iterates over find_stop until all points are
    assigned to clusters.
    :param X: 2-dim array. Contains vector representation for each example.
    :param n_clusters: integer. Number of clusters to be formed.
    :param max_ids_labels: List of arrays. Maximum number of examples per label for each cluster center.
    :param centroids: 2-dim array. Vector representation of centroids.
    :param labels: 2-dim array. gold labels of the examples.
    :return:    final_matrices: list. Contains one calculation matrix for each label.
                    A calculation matrix is a 2-dimensional array. It contains the following information for each
                    example: ID, current_cluster, order measure (distance to current cluster assigned -
                    distance to farthest cluster), desired_cluster, coordinates.
    """

    # create a pandas dataframe with ID and label information for each example
    id_and_class_labels = match_ids_class_labels(labels)

    # create calculation matrices (one calculation matrix per label) and fill it with initial information
    calculation_matrices = set_up_calculation_matrices(X, id_and_class_labels)

    # generate an empty list similar to the calculation matrices
    final_matrices = [None] * len(calculation_matrices)

    # assign points separately for each label
    for j in range(len(calculation_matrices)):

        calculation_matrix = calculation_matrices[j]
        max_ids = max_ids_labels[j]
        centroids_label = centroids

        final_matrix = np.zeros_like(calculation_matrix)
        label_counts = np.zeros_like(max_ids)
        clustering_labels = np.array(range(n_clusters))
        rounds = 0

        # points are assigned. First, the order_measure for each point is calculated.
        # The order measure is the distance to the closest cluster center still open minus the distance to the
        # furthest cluster center still available. The points are sorted by this order measure.
        # The calculation matrix is split at the point where the first cluster
        # reaches its maximum size for the respective label. The points that reached their closest cluster become part
        # of the final matrix. The remaining points stay in the calculation matrix. The distances are again calculated,
        # this time without considering the centroid of the cluster which is already full. Again, points are sorted
        # and assigned. This process ends when the calculation matrix has the length 0, i.e., when all points are
        # assigned.
        while len(calculation_matrix) > 0:
            calculation_matrix[:, (1, 2)] = calculate_order_measure(calculation_matrix[:, 4:], centroids_label,
                                                                clustering_labels)
            calculation_matrix = sort_calculation_matrix(calculation_matrix)
            calculation_matrix, final_matrix, centroids_label, max_ids, label_counts, rounds, clustering_labels = \
                find_stop(calculation_matrix, final_matrix, centroids_label, max_ids, label_counts, rounds,
                          clustering_labels)

        final_matrices[j] = final_matrix

    return final_matrices


def update_centers(final_matrices, centroids):
    """
    combines all "final_matrix" (the final matrix for each respective label) to update the cluster centers.
    :param final_matrices: list containing all individual "final_matrix" (for each label).
    :param centroids: 2-dim array. Vector representation of centroids.
    :return: centroids. 2-dim array. Updated centroids.
    """
    update_matrix = np.concatenate(final_matrices)

    centroids = np.zeros_like(centroids, dtype=np.float64)
    for i in np.unique(update_matrix[:, 1]):
        rows = np.where(update_matrix[:, 1] == i)[0]
        values = update_matrix[:, 4:]
        centroids[int(i)] = np.mean(values[rows, :], axis=0)

    return centroids


def update_distances(calculation_matrix, centroids):
    """
    updates calculation matrix. First calculates the difference in distances from points to their current cluster
    center and to all other cluster centers. Then stores other cluster center ID in the calculation matrix. Cuts of
    all lines with a positive difference, because these points are closer to their current cluster than to the
    other cluster centers. Preparation for swapping, after this step the calculation matrix includes the updated
    information on current cluster center, other cluster center and difference in distances between current and other
    cluster center
    :param calculation_matrix: 2-dim numpy array. calculation matrix with information about all points for a specific
                label.
    :param centroids: 2-dim array. Vector representation of centroids.
    :return: calculation_matrix: 2-dim numpy array with updated distances.
    """

    # calculate distance to current cluster center
    calculation_matrix = calculation_matrix[calculation_matrix[:, 0].argsort()]
    distance_array = cdist(calculation_matrix[:, 4:], centroids, metric="sqeuclidean")
    distance_current_cluster_index = calculation_matrix[:, 1]
    distance_current_cluster = np.zeros_like(calculation_matrix[:, 1])
    for index, values in enumerate(distance_array):
        distance_current_cluster[index] = values[int(distance_current_cluster_index[index])]

    # calculate difference in distances from point to current cluster center and other cluster centers
    differences_matrix = [np.subtract(distance_array[:, c], distance_current_cluster) for c in range(len(centroids))]
    differences_matrix = (np.array(differences_matrix)).T

    # implement the calculation list: the calculation list contains as many copies of the calculation matrix as
    # there are centroids in the data. In each copy of the calculation matrix, the difference in distances between
    # the point and its current cluster center and one other cluster center are stored. The "other" cluster center
    # is stored.
    calculation_list = [None] * len(centroids)
    for i in range(len(centroids)):
        calculation_list[i] = copy.deepcopy(calculation_matrix)

    for c in range(len(centroids)):
        calculation_list[c][:, 2] = differences_matrix.T[c]
        calculation_list[c][:, 3] = (np.repeat(c, len(calculation_list[c][:, 3])))

    # the calculation matrix is a concatenation of all calculation lists, i.e. it contains the differences in distances
    # for each point and each centroid
    calculation_matrix = np.concatenate(calculation_list)

    # sort by difference in distances
    calculation_matrix = calculation_matrix[calculation_matrix[:, 2].argsort()]

    # cut of all points with positive difference in distances --> those points are closer to their current cluster
    # center, than to the other cluster center.
    # Why do we keep points with distance 0? Each point has to be present in the calculation matrix at least one
    # time.
    calculation_matrix = calculation_matrix[[calculation_matrix[:, 2] <= 0][0], :]

    return calculation_matrix


def cut_non_transfers(calculation_matrix):
    """
    calculation_matrix still contains one entry for each point with difference in distances 0. (calculation of
    difference in distances two times for current cluster center). If a point only appears once with distance 0, it
    cannot be a swap candidate, because the only swap yielding an increase in the overall cluster-internal variance
    would be the swap from the current cluster center to the current cluster center. These points become so-called
    non-transfers. For the remaining points, all all entries with a difference in distance that are not 0 become
    transfer points.
    :param calculation_matrix: 2-dim numpy array. calculation matrix with information about all points for a specific
                label.
    :return:    non_transfers: 2-dim numpy array. Non candidate points for swapping.
                transfers: 2-dim numpy array. Candidate points for swapping.
    """

    # extract which points appear only once in calculation matrix --> these points become non-transfers
    occurence_ids = np.unique(calculation_matrix[:, 0], return_counts=True)
    index_single = np.where(occurence_ids[1] == 1)
    id_single = occurence_ids[0][index_single[0]]
    non_transfer_ids = [calculation_matrix[row, 0] in id_single for row in range(len(calculation_matrix))]
    non_transfers = calculation_matrix[non_transfer_ids, :]
    # all other points become transfer candidates, except if the calculated difference in distances is 0 (as this would
    # imply a swap from the current cluster center to the current cluster center
    transfer_ids = np.where(calculation_matrix[:, 2] != 0)[0]
    transfers = calculation_matrix[transfer_ids]

    return non_transfers, transfers


def swap(transfers):
    """
    performs one-on-one swapping of swap candidates. In detail, the method iterates through the transfer matrix in the
    following way. It looks at the first example in the matrix first and keeps this information. It iterates through
    the remaining points in the calculation matrix and if it finds one, that has the reverse combination of current
    cluster center and desired cluster center, it sets the swap indicator of these two instances to 1, indicating that
    they can be swapped. As each point currently can appear several times in the calculation matrix, the id_stepper is
    set to 1 for all instances of the points that have just swapped. This ensures that each point can swap only
    one time when the method is called.
    :param transfers: 2-dim numpy array. Candidate points for swapping.
    :return:    transfers: 2-dim numpy array. updated cluster assignment. each point appears only once.
                swap_indicator: 1-dim np.array: contains the number 1 if a swap was performed for the respective point.
                    0 if no swap was performed. Serves for evaluating, whether there are still swaps happening.
    """
    swap_indicator = np.zeros(len(transfers))
    id_stepper = np.zeros(len(transfers))
    for index, values in enumerate(transfers):
        # point cannot swap if it has already swapped
        if (swap_indicator[index] == 1 or id_stepper[index] == 1):
            pass
        else:
            for i in range(index, len(transfers)):
                # check if there is a point with the reverse combination of current cluster and desired cluster
                if (swap_indicator[i]==0 and id_stepper[i]==0 and transfers[index, 1]==transfers[i, 3] and transfers[index, 3] == transfers[i, 1]):
                    swap_indicator[i] = 1
                    swap_indicator[index] = 1
                    id_stepper[np.where(transfers[:, 0] == transfers[i,0])[0]] = 1
                    id_stepper[np.where(transfers[:, 0] == transfers[index, 0])[0]] = 1
                    break

    # perform swap = change desired cluster and current cluster for points that can swap
    transfers[:, 1] = np.where(swap_indicator == 1, transfers[:, 3], transfers[:, 1])
    # for all ids that were swapped, remove duplicate instances
    duplicates_swaps = [all((swap_indicator[i] == 0, id_stepper[i] == 1)) for i in range(len(swap_indicator))]
    transfers = transfers[np.invert(duplicates_swaps), :]

    # find indexes for unique IDS
    unique_indexes = np.unique(transfers[:, 0], return_index=True)[1]
    #remove duplicates from non-swaps
    transfers = transfers[unique_indexes.astype(int), :]

    return transfers, swap_indicator


def unite_data(transfers, non_transfers):
    """
    unite transfers and non_transfers to create one calculation matrix.
    :param transfers: transfers: 2-dim numpy array. updated cluster assignment.
    :param non_transfers: 2-dim numpy array. Points that were not candidates for swapping.
    :return: calculation_matrix. 2-dim array. Each point appears only once.
    """
    calculation_matrix = np.concatenate((transfers, non_transfers), axis=0)
    calculation_matrix = calculation_matrix[calculation_matrix[:, 0].argsort()]
    return calculation_matrix


def calculate_cluster_inertia(centroids, calculation_matrix):
    """
    calculates sum of average squared euclidean distanecs between each cluster center and the associated points.
    :param centroids: centroids: 2-dim array. Vector representation of centroids.
    :param calculation_matrix: 2-dim array with point information.
    :return: cluster_inertia. Float. Sum of average euclidean divergences between cluster centers and associated points.
    """
    # calculate euclidean distance between each ID and centroids
    distance_array = cdist(calculation_matrix[:, 4:], centroids, metric="sqeuclidean")

    # extract which centroid ID belongs to
    cluster = calculation_matrix[:, 1]

    distances = [distance_array[i, int(cluster[i])] for i in range(len(distance_array))]

    cluster_inertia = [None] * len(centroids)

    for clu in cluster:
        index = np.where(cluster == clu)[0]
        distances_cluster = np.take(distances, index)
        cluster_inertia[int(clu)] = np.mean(distances_cluster)

    cluster_inertia = sum(cluster_inertia)

    return cluster_inertia


def kmeans_distribution_single_round(current_round, initializations, X, labels, n_clusters, max_ids_labels,
                                     init_kmeans, max_iter):
    """
    Performs a single round of the size and distribution sensitive K-Means algorithm, i.e., initializes one partition
    with a pre-specified random seed.
    :param current_round: Current round in initializing the algorithm.
    :param initializations: 1-dim array. Contains all random seeds for all rounds of initializaing the algorithm.
    :param X: 2-dim array. Contains vector representation for each example.
    :param labels: 2-dim array. gold labels of the examples.
    :param n_clusters: integer. Number of clusters to be formed.
    :param max_ids_labels: List of arrays. Each array describes the distribution of one of the labels over the
                                different clusters. It is very important that the total number of examples distributed
                                over the different labels matches the total number of examples of the respective labels.
                                Otherwise, the algorithm will produce an error!
                                Example: [np.array([1,1,1], np.array([1,2,3])]
                                Here, there are 3 clusters formed. Cluster 1 can incorporate 1 example from the first
                                label and 1 example from the second label. Cluster 2 can incorporate 1 example from the
                                first label and two examples from the second label. Cluster 3 can incorporate 1 example
                                from the first label and 3 examples from the second label.
    :param init_kmeans: string. Initialization method to be applied.
    :param max_iter: integer. Maximum number of rounds in reassigning the points in the update step.
    :return:    inertia: float. Sum of average distances from points their assigned cluster center.
                centroids: 2-dim array. Vector Representations of cluster centroids.
                cluster_labels: 1-dim array. Labels assigned to the different examples.
    """

    ##### INITIAL ASSIGNMENT STEP #####

    # centroids are set according to _init_centroids method from class cluster.KMeans
    centroids = _init_centroids(X, n_clusters, init_kmeans, random_state=initializations[current_round])

    # There is a calculation matrix for each label existing within the data set. Each calculation matrix contains
    # all data points for the respecitve label. The calculation matrices for the individual labels are stored in a list,
    # which is called "calculation_matrices". Each calculation matrix includes the following information:
    # example ID, current_cluster (the datapoint belongs to), distance, desired_cluster (the data point would like
    # to become part of), coordinates. The calculations performed throughout the algorithm are stored in the calculation
    # matrices.

    # each point is assigned to one cluster center
    calculation_matrices = first_assignment(X, n_clusters, max_ids_labels, centroids, labels)

    # uncomment these commands if you would like to extract the initial clustering (without any swaps happening)
    # final_result = np.concatenate(calculation_matrices)
    # final_result = final_result[final_result[:,0].argsort()]

    ##### UPDATE STEP #####

    # Flag is set to True if no more swaps are happening
    stop_execution = False

    # swaps will be executed for the maximum number of max_iter rounds
    for j in range(max_iter):

        # recalculate cluster centers
        centroids = update_centers(calculation_matrices, centroids)

        # if there happened no more swaps in the previous round, stop execution here (after updating the cluster
        # centers)
        if stop_execution:
            break

        # prepare swapping
        swap_indicators = [None] * len(calculation_matrices)
        total_transfers = [None] * len(calculation_matrices)
        total_non_transfers = [None] * len(calculation_matrices)

        # swapping is performed for each label separately
        for z in range(len(calculation_matrices)):

            # extract the calculation matrix for the respective label
            calculation_matrix = calculation_matrices[z]
            calculation_matrix = update_distances(calculation_matrix, centroids)

            # points that are closer to another cluster center than to their current cluster center become
            # "transfers". "non_transfers" are closer to the current cluster center than to other cluster centers
            # and thus are not considered for swapping.
            non_transfers, transfers = cut_non_transfers(calculation_matrix)
            transfers, swap_indicator = swap(transfers)
            calculation_matrix = unite_data(transfers, non_transfers)

            # store result for this label
            calculation_matrices[z] = calculation_matrix
            swap_indicators[z] = swap_indicator
            total_transfers[z] = transfers
            total_non_transfers[z] = non_transfers

            # if all labels were considered in the current round and no swaps happened, prepare output
            length = len(calculation_matrices)
            if (j > 0 and (z+1) == len(calculation_matrices)) or j == max_iter:
                if np.count_nonzero(np.concatenate(swap_indicators)) == 0:
                    final_result = np.concatenate(calculation_matrices)
                    # sort, so that labels are output in the same order that vectors were input to the method
                    final_result = final_result[final_result[:, 0].argsort()]
                    print('calculations stopped at {}'.format(j))
                    stop_execution = True

    inertia = calculate_cluster_inertia(centroids, final_result)
    cluster_labels = final_result[:, 1]

    return inertia, centroids, cluster_labels


class KMeansDistribution(cluster.KMeans):
    """
    Class implementing the size and distribution sensitive K-Means algorithm.
    The size and distribution sensitive K-Means algorithm implements a pre-specified number of clusters (n_clusters)
    with a pre-defined size and label distribution (defined by max_ids_labels). Its basic building blocks are an
    initial assignment step, where all examples are assigned towards one cluster center, and an updating step where
    points are reassigned. The class inherits from the class cluster.KMeans from the Python sklearn package.

    ---
    Attributes
    ---
    n_clusters: integer. Number of clusters to be formed.
    init: string. Initialization method to be applied.
    n_init: integer. Number of initializations.
    max_iter: integer. Maximum number of rounds in reassigning the points in the update step.
    random_state: integer. Random state for initializing first assignment.
    max_ids_labels: List of arrays. Maximum number of examples per label for each cluster center.
    cluster_centers: two-dimensional array. Vector representation of cluster centers.
    clustering_labels: 1-dim array. Cluster labels assigned, sorted the same way that the vector inputs for the
                            different examples are sorted.

    ---
    Methods
    ---
    fit: conduct clustering.

    """

    def __init__(self, n_clusters=5, init='k-means++', n_init=8, max_iter=100, random_state=110,
                 max_ids_labels=None):
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)
        """
        Initializes an instance of the class KMeansDistribution. Inherits from class sklearn.cluster.KMeans
        :param n_clusters: integer. Number of clusters to be formed.
        :param init: string. Initialization method to be applied.
        :param n_init: integer. Number of initializations.
        :param max_iter: integer. Maximum number of rounds in reassigning the points in the update step.
        :param random_state: integer. Random state for initializing the first assignement.
        :param max_ids_labels: List of arrays. Each array describes the distribution of one of the labels over the
                                different clusters. It is very important that the total number of examples distributed
                                over the different labels matches the total number of examples of the respective labels.
                                Otherwise, the algorithm will produce an error!
                                Example: [np.array([1,1,1], np.array([1,2,3])]
                                Here, there are 3 clusters formed. Cluster 1 can incorporate 1 example from the first
                                label and 1 example from the second label. Cluster 2 can incorporate 1 example from the
                                first label and two examples from the second label. Cluster 3 can incorporate 1 example
                                from the first label and 3 examples from the second label.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_ids_labels = max_ids_labels

        # these attributes are attained through performing the "fit" method
        self.cluster_centers = None
        self.clustering_labels = None


    def fit(self, X, gold_labels, n_jobs):
        """
        Initializes the size and distribution sensitive K-Means algorithm multiple times with different random seeds.
        Selects the partition with least sum of average Euclidean distances from points to cluster centers.
        :param X: 2-dim array. Contains vector representation for each example.
        :param gold_labels: 2-dim array. gold labels of the examples.
        :param n_jobs: integer. Number of parallel jobs to execute during parallelization.
        :return: self.
        """
        # set seed and extract series of random numbers for n_init initializations of algorithm
        random.seed(self.random_state)
        initializations = random.sample(range(1,100000), self.n_init)

        # each initialization needs to be conducted with the correct seed. To keep track of which initialization
        # currently should be used, create all_inits variable.
        all_inits = list(range(self.n_init))

        # execute n_init rounds with different initializations.
        all_partitions = Parallel(n_jobs=n_jobs, verbose=1)(delayed(kmeans_distribution_single_round)
                                                            (current_round=round_init,
                                                            initializations=initializations,
                                                            X=X,
                                                            labels=gold_labels,
                                                            n_clusters=self.n_clusters,
                                                            max_ids_labels=self.max_ids_labels,
                                                            init_kmeans=self.init,
                                                            max_iter=self.max_iter)
                                                            for round_init in all_inits)

        # collect results from all rounds. all_transfers are all points which would have liked to
        # perform a swap, but could not perform swaps anymore (i.e., all points which were not assigned to
        # their closest cluster center)
        inertia, cluster_centers, clustering_labels = zip(*all_partitions)

        # report result with smallest average Euclidean distance from points to cluster center
        minimum = np.argmin(inertia)
        self.cluster_centers_ = cluster_centers[minimum]
        self.clustering_labels_ = clustering_labels[minimum]

        return self


def get_max_of_bin(value, bins):
    max_val = 0
    for bin_max in bins:
        max_val = bin_max
        if value < bin_max:
            break
    return max_val


def generate_overview_stats(text_data, labels, num_bins=10):
    """
    Generates number_ids, length_text, unique_labels and label_text_length from text and label information.
    :param text_data: list of strings. Contains text data for all examples.
    :param labels: 2-dimensional array. Contains gold labels for each example.
    :param num?bins: number of buckets/bins to group input text lengths
    :return:    number_ids: integer. Number of examples.
                length_text: 1-dim array. Number of tokens for each example.
                unique_labels. tuple. First array describes labels occuring, second array describes number of occurence
                    of the different labels.
                label_text_length: pandas dataframe. Columns "label" and "length_text" (measured in tokens).
    """
    number_ids = len(text_data)
    length_text = [len(string.split()) for string in text_data]
    # prepare bins
    max_text_length = max(length_text)
    bin_size = int(max_text_length/num_bins)
    bins = np.array([(x+1)*bin_size for x in range(num_bins)])
    length_text_binned = [get_max_of_bin(v, bins) for v in length_text]
    
    unique_labels = np.unique(labels, return_counts=True)
    idx2label = {i : l for i, l in enumerate(unique_labels[0])}
    label2idx = {l : i for i, l in enumerate(unique_labels[0])}
    
    # with numeric labels (indices)
    numeric_labels = [label2idx[l] for l in labels[0]]
    label_text_length = pd.DataFrame((numeric_labels), columns=['label'])
    label_text_length['length_text'] = length_text
    label_text_length['length_text_binned'] = length_text_binned
                                             
    return number_ids, length_text, unique_labels, label_text_length, idx2label, label2idx


def generate_tokens_and_frequencies(text_data, rank):
    """
    Extracts the most <rank> most frequent tokens with their respective frequencies.
    :param text_data: list of strings. Contains text data for all examples.
    :param rank: integer. Number of tokens and their respective frequencies to be stored.
    :return:    tokens: tuple. Contains <rank> most frequent tokens sorted by frequency.
                frequencies: tuple. number of (absolute) occurence for <rank> most frequent tokens.
                full_text_string: string. Contains the text data from all examples merged in one string.
    """
    full_text_string = ' '.join(text_data)
    token_split = full_text_string.split()
    token_counter = collections.Counter(token_split)
    most_frequent = token_counter.most_common(rank)
    tokens, frequencies = zip(*most_frequent)

    return tokens, frequencies, full_text_string


def create_sentence_vectors(text_data, model):
    """
    Creates sentence vectors by averaging the word vectors for each sentence.
    :param text_data: list. preprocessed input data, contains text string for each example.
    :param model: (gensim) model. Word2vec model that should be used for creating word vectors.
    :return:    sentence vectors: dictionary. Key: ID for each example. Value: example vector representation.
                exceptions. Integer. Number of exceptions. An exception means the model did not know all
                words in a sentence and thus had to leave out words in order to transform a sentence in a vector
                representation.
    """
    sentence_vectors = {}
    exceptions = 0
    
    vector_size = len(model[list(model.vocab.keys())[0]])  # retrieve length of vector

    for instance, text in enumerate(text_data):
        # returns dictionary with (average) sentence vectors. Key: ID in text_dataset
        try:
            vectors = [model[x] for x in text.split(' ')]
            sentence_vector = sum(vectors) / len(vectors)
        except:
            # exception if model does not know all words in a sentence
            sentence_vector = []
            vectors = []
            for word in text.split(' '):
                try:
                    # try to leave out the words the model doesn't know and obtain a sentence vector by averaging over
                    # the rest of the words
                    vectors.append(model[word])
                except:
                    pass
            try:
                sentence_vector = sum(vectors) / len(vectors)
            except:
                # if it is not possible to use any word of the sentence for creating the sentence vector, sentence
                # vector is left empty, using zeros here
                sentence_vector = [0 for i in range(vector_size)]
            exceptions += 1

        key = instance
        sentence_vectors[key] = sentence_vector

    return sentence_vectors, exceptions


def perform_pca(vectors, n_components):
    """
    reduces the number of dimensions by principal component analysis.
    :param vectors: 2-dim array. Vectors for which dimensionality reduction is desired. need to be centered and scaled!
    :param n_components: integer. Number of dimensions after reduction.
    :return:    (1) sentence_vectors: 2-dim array. vectors with reduced dimensionality
                (2) explained_variance_ratio. 1-dim array. percentage of variance in original data that is explained by
                the selected principal components
                (3) explained_variance: 1-dim array. absolute size of eigenvalues belonging to the selected principal
                components.
    """
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_
    reduced_vectors = pca.fit_transform(vectors)

    return reduced_vectors, explained_variance_ratio, explained_variance

def generate_default_label_distribution(gold_labels, num_clusters):
    """
    implements the same distribution that is present in the full data set for every cluster
    :param text_input_data: pandas dataframe. Contains label information in second column.
    :param num_clusters: integer. Number of clusters which should be formed in the data.
    :return: max_ids_labels. 2-dim numpy array. Contains information on how many examples for each label should be assigned
                to the different clusters (minimum and maximum number of instances per cluster)
    """
    labels = gold_labels
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label2idx = {}
    
    for i, l in enumerate(unique_labels):
        label2idx[l] = i
        
    for label in unique_labels:

        ids_per_cluster = (label_counts[label2idx[label]] / num_clusters)
        ceils = int(round(ids_per_cluster % 1, 2) * num_clusters)
        floors = num_clusters - ceils

        max_ids_label = np.array([[int(np.ceil(ids_per_cluster))] * ceils + [int(np.floor(ids_per_cluster))] * floors])

        if label2idx[label] == 0:
            max_ids_labels = max_ids_label

        elif label2idx[label] > 0:
            max_ids_labels = np.concatenate((max_ids_labels, max_ids_label))

    return max_ids_labels


def perform_kmeans(vectors, num_clusters):
    """
    Performs regular K-Means clustering
    :param vectors: 2-dim array. sentence vectors, centered and scaled.
    :param num_clusters: integer. Number of clusters which should be explored.
    :return:    (1) clustering_labels. 1-dim array. Cluster IDs assigned to the examples.
                (2) centroids. 2-dim array. Centroids associated to the clusters.
    """

    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=110)
    kmeans.fit(vectors)

    clustering_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return clustering_labels, centroids


def perform_kmeans_size(vectors, num_clusters, n_jobs, max_ids_labels=None):
    """
    Performs same_size K-Means algorithm by calling same size K-Means algorithm and assigning the same (fake) labek
    to all datapoints.
    :param vectors: 2-dim array. sentence vectors, centered and scaled.
    :param num_clusters: integer. Number of clusters which should be explored.
    :param n_jobs: integer. number of workers for parallelization.
    :param max_ids_labels: list. maximum number of ids per label in each cluster.
                           example: [np.array([1,1,1]), np.array([2,2,2])] means that from the first label,
                           there should be one example in each of the three clusters, from the second label there
                           should be three examples in each of the three clusters. Attention: the number of examples
                           assigned to the different clusters must exactly match the number of examples in the whole
                           dataset.
    :return:    (1) clustering_labels. 1-dim array. Cluster IDs assigned to the examples.
                (2) centroids. 2-dim array. Centroids associated to the clusters.
    """

    # create 'fake' gold labels: Same Size K-Means algorithm is executed when calling Size and Distribution Sensitive
    # algorithm and all points have the same label


    kmeans_size_distribution = KMeansDistribution(n_clusters=num_clusters, random_state=10,
                                                  max_ids_labels=max_ids_labels)

    gold_labels = np.array([np.repeat(0, len(vectors))])

    kmeans_size_distribution.fit(vectors, gold_labels, n_jobs)

    clustering_labels = kmeans_size_distribution.clustering_labels_
    centroids = kmeans_size_distribution.cluster_centers_

    return clustering_labels, centroids


def perform_kmeans_size_distribution(vectors, num_clusters, gold_labels, n_jobs, max_ids_labels=None):
    """
    Performs our size and distribution sensitive K-Means algorithm.
    :param vectors: 2-dim array. sentence vectors, centered and scaled.
    :param num_clusters: integer. Number of clusters to be created.
    :param gold_labels: 2-dim array. gold labels of the examples.
    :param n_jobs: integer. number of workers for parallelization.
    :param max_ids_labels: list. maximum number of ids per label in each cluster.
                           example: [np.array([1,1,1]), np.array([2,2,2])] means that from the first label,
                           there should be one example in each of the three clusters, from the second label there
                           should be three examples in each of the three clusters. Attention: the number of examples
                           assigned to the different clusters must exactly match the number of examples in the whole
                           dataset.
    :return:    (1) clustering_labels. 1-dim array. Cluster IDs assigned to the examples.
                (2) centroids. 2-dim array. Centroids associated to the clusters.
    """

    kmeans_size_distribution = KMeansDistribution(n_clusters=num_clusters, random_state=10,
                                                 max_ids_labels=max_ids_labels)
    
    # map gold labels to integers
    label2idx = {}
    for i, l in enumerate(np.unique(gold_labels)):
        label2idx[l] = i
        
    gold_labels = [[label2idx[l] for l in gold_labels[0]]]
    
    kmeans_size_distribution.fit(vectors, gold_labels, n_jobs)

    clustering_labels = kmeans_size_distribution.clustering_labels_
    centroids = kmeans_size_distribution.cluster_centers_

    return clustering_labels, centroids


def create_random_datasplits(vectors, num_folds):
    """
    Examples are assigned randomly to the different data folds.
    :param vectors: 2-dim array. sentence vectors.
    :param num_folds: integer. Number of data folds to be produced.
    :return:    fold_labels: 1-dim array. Data folds the examples belong to.
    """
    # splits should have same size --> determine how many ids are in data set
    num_ids = len(vectors)
    ids_per_fold = num_ids/num_folds
    fold_ids = list(range(0, num_folds))
    fold_labels = np.repeat(fold_ids, np.ceil(ids_per_fold))
    random.shuffle(fold_labels)
    fold_labels = fold_labels[0: num_ids]

    return fold_labels
