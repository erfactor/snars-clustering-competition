import networkx as nx
import numpy as np
import sklearn.preprocessing
from scipy.sparse import isspmatrix, dok_matrix, csc_matrix


def normalize(m):
    return sklearn.preprocessing.normalize(m, norm="l1", axis=0)


def sparse_allclose(a, b, rtol=1e-5, atol=1e-8):
    c = np.abs(a - b) - rtol * np.abs(b)
    return c.max() <= atol


def add_self_loops(m):
    shape = m.shape
    new_m = m.todok() if isspmatrix(m) else m.copy()

    for i in range(shape[0]):
        new_m[i, i] = 1

    if isspmatrix(m):
        return new_m.tocsc()

    return new_m


def prune(m):
    threshold = 0.001
    if isspmatrix(m):
        pruned = dok_matrix(m.shape)
        pruned[m >= threshold] = m[m >= threshold]
        pruned = pruned.tocsc()
    else:
        pruned = m.copy()
        pruned[pruned < threshold] = 0

    column_count = m.shape[1]
    rows = m.argmax(axis=0).reshape((column_count,))
    columns = np.arange(column_count)
    pruned[rows, columns] = m[rows, columns]

    return pruned


def are_matrices_similar(m1, m2):
    if isspmatrix(m1) or isspmatrix(m2):
        return sparse_allclose(m1, m2)

    return np.allclose(m1, m2)


def step(m):
    expansion = 3
    inflation = 3
    m = expand(m, expansion)
    m = inflate(m, inflation)
    return m


def expand(m, power):
    if isspmatrix(m):
        return m ** power
    return np.linalg.matrix_power(m, power)


def inflate(m, power):
    if isspmatrix(m):
        return normalize(m.power(power))
    return normalize(np.power(m, power))


def find_clusters(network, iterations=100):
    m = nx.to_scipy_sparse_matrix(network)
    m = add_self_loops(m)
    m = normalize(m)

    for i in range(iterations):
        last_matrix = m.copy()
        m = step(m)
        m = prune(m)
        if are_matrices_similar(m, last_matrix):
            break

    return matrix_to_clusters(m)


def matrix_to_clusters(m):
    if not isspmatrix(m):
        m = csc_matrix(m)

    attractors = m.diagonal().nonzero()[0]
    result = set()
    for attractor in attractors:
        cluster = tuple(m.getrow(attractor).nonzero()[1].tolist())
        result.add(cluster)

    return sorted(list(result))
