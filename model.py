import numpy as np
import random
from scipy.io import arff
import math
from matplotlib import pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score


def load_arff(file_path):
    data, meta = arff.loadarff(file_path)
    x = []
    y = []
    for w in range(len(data)):
        x.append([])
        for k in range(len(data[0])):
            if k == (len(data[0]) - 1):
                y.append(data[w][k])
            else:
                x[w].append(data[w][k])
    classes = list(set(y))
    return x, y, classes


def split_data(x, alfa):
    idx = np.random.permutation(len(x))
    xu = [x[i] for i in idx[0:int(np.floor(len(x) * alfa) + 1)]]
    return xu


def dp(X, C, p):
    result = {}
    for centre_index in range(len(C)):
        result[centre_index] = []
        for sample in X:
            temp = [pow((abs(sample[i] - C[centre_index][i])), p) for i in range(len(C[0]))]
            dist = pow(sum(temp), 1 / p)
            result[centre_index].append(dist)
    return result


def dm(X, C):
    result = {}
    A = np.cov(np.array(X).transpose())
    for centre_index in range(len(C)):
        result[centre_index] = []
        for sample in X:
            x_c = np.array(sample) - np.array(C[centre_index])
            dist = math.sqrt(np.dot(np.dot(x_c, inv(A)), x_c.transpose()))
            result[centre_index].append(dist)
    return result


def random_select(X, k):
    """
    Return k randomly chosen points from x
    """
    rand_indexes = [random.randint(1, len(X)) for i in range(k)]
    return [X[i] for i in rand_indexes]


def k_means(X, k, p=2, dist_function="Euclidean"):
    C = random_select(X, k)

    while 1:
        if dist_function == "Euclidean":
            dist = dp(X, C, p)
        elif dist_function == "Mahalonobis":
            dist = dm(X, C)

        CX = {}
        for i in range(k):
            CX[i] = []

        for i in range(len(X)):
            temp_dist = dist[0][i]
            temp_centre = 0
            for centre in range(1, k):
                if dist[centre][i] < temp_dist:
                    temp_centre = centre
                    temp_dist = dist[centre][i]
            CX[temp_centre].append(i)
        C_new = []

        for centre in range(k):
            temp_centre = []
            for dim in range(len(X[0])):
                temp_centre.append(np.mean([X[i][dim] for i in CX[centre]]))
            C_new.append(temp_centre)

        if C_new == C:
            break
        else:
            C = C_new
    return C, CX


def clustering_error(X, C, CX, dist_function="Euclidean"):
    total_error = 0
    if dist_function == "Euclidean":
        distances = dp(X, C, 2)
    elif dist_function == "Mahalonobis":
        distances = dm(X, C)

    for cluster in range(len(C)):
        for point in CX[cluster]:
            total_error += distances[cluster][point]
    return total_error


def plot_clusters_2d(CX, X):
    for cluster in range(len(CX)):
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        for i in CX[cluster]:
            plt.scatter(X[i][0], X[i][1], s=4, c=color)

    plt.title("Wizulizacja danych w rzucie na 2 losowe zmienne - algorytm k-środków")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def plot_clusters_3d(CX, X):
    fig = plt.figure()
    ax = Axes3D(fig)

    for cluster in range(len(CX)):
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        for i in CX[cluster]:
            ax.scatter(X[i][0], X[i][1], X[i][2], s=4, c=color)

    plt.title("Wizulizacja danych w rzucie na 3 losowe zmienne - algorytm k-środków")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.zaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(True)

    plt.show()


def plot_data_set_2d(x, y):
    classes = list(set(y))
    colors = ["#%06x" % random.randint(0, 0xFFFFFF) for i in range(len(classes))]

    for i in range(len(x)):
        idx = classes.index(y[i])
        plt.scatter(x[i][0], x[i][1], s=4, c=colors[idx])

    plt.title("Wizulizacja danych w rzucie na 2 losowe zmienne")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def plot_data_set_3d(x, y):
    fig = plt.figure()
    ax = Axes3D(fig)

    classes = list(set(y))
    colors = ["#%06x" % random.randint(0, 0xFFFFFF) for i in range(len(classes))]

    for i in range(len(x)):
        idx = classes.index(y[i])
        ax.scatter(x[i][0], x[i][1], x[i][2], s=4, c=colors[idx])

    plt.title("Wizulizacja danych w rzucie na 3 losowe zmienne")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.zaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(True)

    plt.show()


def find_corresponding_classes(CX, Y):
    CX_new = {}
    for i in range(len(CX)):
        classes = [Y[i] for i in CX[i]]
        most_common = max(set(classes), key=classes.count)
        CX_new[most_common] = CX[i]
    return CX_new


def kmeans_accuracy(CX, Y):
    num_of_samples = len(Y)
    incorretly_classified = 0
    keys = list(CX.keys())

    for key in keys:
        for i in CX[key]:
            if Y[i] != key:
                incorretly_classified += 1

    return (num_of_samples - incorretly_classified) / num_of_samples


def measure_time(x):
    time_measurements = []

    for i in range(1, 20):
        x_temp = split_data(x, i / 20)

        start = time.time()
        C, Cx = k_means(x_temp, 2)
        end = time.time()
        time_measurements.append(end - start)
    return time_measurements
