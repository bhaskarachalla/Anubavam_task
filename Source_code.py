import numpy as np
from numpy.linalg import norm
import csv

class matrix:
    def __init__(self, filename=None, array_2d=None):  # This class allows initialization
                                                      # with a CSV filename (to load data from CSV)
                                                      # or with a 2D array.
        # Start array_2d as an empty NumPy array
        self.array_2d = np.array([])

        if filename:
            self.load_from_csv(filename)
        # If array_2d is provided, use it
        elif array_2d is not None:  # Check if array_2d isn't None
            self.array_2d = array_2d

    # 1) Loading CSV file
    def load_from_csv(self, filename):
        # Read CSV and load data into array_2d
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            data = [list(map(float, row)) for row in reader]
            self.array_2d = np.array(data)

    # 2) Make the matrix standard
    def standardize(self):
        for j in range(self.array_2d.shape[1]):
            col = self.array_2d[:, j]
            self.array_2d[:, j] = (col - np.mean(col)) / (np.max(col) - np.min(col))

    # 3) Figure out Euclidean distance
    def get_distance(self, other_matrix, row_i):
        diff = self.array_2d[row_i, :] - other_matrix.array_2d
        dist = np.sqrt(np.sum(diff**2, axis=1))
        return dist.reshape(-1, 1)

    # 4) Find the weighted Euclidean distance
    def get_weighted_distance(self, other_matrix, weights, row_i):
        diff = self.array_2d[row_i, :] - other_matrix.array_2d
        weighted_diff = weights.array_2d * (diff**2)
        dist = np.sqrt(np.sum(weighted_diff, axis=1))
        return dist.reshape(-1, 1)

    # 5) Count how often unique values show up (for a single-column matrix)
    def get_count_frequency(self):
        if self.array_2d.shape[1] != 1:
            return 0
        return dict(zip(*np.unique(self.array_2d, return_counts=True)))

# Function to get starting random weights
def get_initial_weights(m):
    weights = np.random.rand(1, m)
    return weights / np.sum(weights)

# Function to get centroids
def get_centroids(data, S, K):
    centroids = np.zeros((K, data.array_2d.shape[1]))
    for k in range(K):
        group_data = data.array_2d[S.array_2d[:, 0] == k]
        if group_data.shape[0] > 0:
            centroids[k, :] = np.mean(group_data, axis=0)
    return matrix(array_2d=centroids)

# Function to calculate separation within clusters
def get_separation_within(data, centroids, S, K):
    separation_within = np.zeros((1, data.array_2d.shape[1]))
    for j in range(data.array_2d.shape[1]):
        for k in range(K):
            group_data = data.array_2d[S.array_2d[:, 0] == k]
            for i in range(group_data.shape[0]):
                separation_within[0, j] += norm(group_data[i, j] - centroids.array_2d[k, j])**2
    return separation_within

# Function to figure out the distance between clusters
def get_seperation_between(data, centroids, S, K):
    gaps_between = np.zeros((1, data.array_2d.shape[1]))
    for j in range(data.array_2d.shape[1]):
        for k in range(K):
            gaps_between[0, j] += norm(centroids.array_2d[k, j] - np.mean(data.array_2d[:, j]))**2
    return gaps_between

# Function to make groups
def get_groups(data, K):
    S = matrix(array_2d=np.zeros((data.array_2d.shape[0], 1)))
    # Make sure centroids_matrix has the right size: K rows and same number of columns as data
    centroids = data.array_2d[np.random.choice(data.array_2d.shape[0], K, replace=False), :]
    centroids_matrix = matrix(array_2d=centroids)
    weights = get_initial_weights(data.array_2d.shape[1])
    old_S = None

    while not np.array_equal(S.array_2d, old_S):
        old_S = np.copy(S.array_2d)
        for i in range(data.array_2d.shape[0]):
            # Make sure 'self' is the data point and 'other_matrix' are the centroids in get_weighted_distance
            distances = data.get_weighted_distance(centroids_matrix, matrix(array_2d=weights), i)
            S.array_2d[i, 0] = np.argmin(distances)
        for k in range(K):
            for j in range(data.array_2d.shape[1]):
                group_data = data.array_2d[S.array_2d[:, 0] == k]
                if group_data.shape[0] > 0:
                    centroids_matrix.array_2d[k, j] = np.mean(group_data[:, j])

    return S

# Function to calculate new weights
def get_new_weights(data, centroids, old_weights, S, K):
    a = get_separation_within(data, centroids, S, K)
    b = get_seperation_between(data, centroids, S, K)
    new_weights = old_weights * (a / b)**0.5
    return new_weights / np.sum(new_weights)

# Function to run test
def run_test():
    m = matrix('Anubavam_dataset.csv')
    for k in range(2, 11):
        for i in range(20):
            S = get_groups(m, k)
            print(str(k) + '=' + str(S.get_count_frequency()))

run_test()
