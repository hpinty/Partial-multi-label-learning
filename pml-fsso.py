"""
<Partial Multi-Label Feature Selection
via Subspace Optimization>
"""


import numpy as np
from skfeature.utility.construct_W import construct_W
from numpy import linalg as LA
from numpy.random import seed

eps = 2.2204e-16


def first_idea_process(X, Y, alpha, beta, gamma, theta):
    num, dim = X.shape
    num, label_num = Y.shape
    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 5, 'weight_mode': 'heat_kernel', 't': 1.0}
    Sx = construct_W(X, **options)
    Sx = Sx.A
    Ax = np.diag(np.sum(Sx, 0))
    Lx = Ax - Sx

    k = int(np.ceil(dim * 0.8))

    seed(2)
    U = np.random.rand(num, k)
    V = np.random.rand(k, label_num)
    M = np.random.rand(k, dim)

    iter = 0
    obj = []
    obji = 1

    while 1:
        temp_v = np.dot(M.T, V)
        Btmp = np.sqrt(np.sum(np.multiply(temp_v, temp_v), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)

        V = np.multiply(V, np.true_divide(np.dot(np.dot(M, X.T), Y) + alpha * np.dot(U.T, Y),
                                          np.dot(np.dot(np.dot(M, X.T), X), temp_v) + alpha * np.dot(np.dot(U.T, U),
                                                                                                     V) + 2 * theta * np.dot(
                                              np.dot(M, D), temp_v) + eps))

        U = np.multiply(U, np.true_divide(alpha * np.dot(Y, V.T) + beta * np.dot(X, M.T) + gamma * np.dot(Sx, U),
                                          alpha * np.dot(np.dot(U, V), V.T) + beta * np.dot(np.dot(U, M),
                                                                                            M.T) + gamma * np.dot(Ax,
                                                                                                                  U) + eps))

        M = np.multiply(M, np.true_divide(np.dot(V, np.dot(Y.T, X)) + beta * np.dot(U.T, X),
                                          np.dot(np.dot(np.dot(V, temp_v.T), X.T), X) + beta * np.dot(np.dot(U.T, U),
                                                                                                      M) + 2 * theta * np.dot(
                                              np.dot(np.dot(V, V.T), M), D) + eps))

        # test
        temp_v = np.dot(M.T, V)
        part1 = pow(LA.norm(np.dot(X, temp_v) - Y, 'fro'), 2)
        part2 = alpha * pow(LA.norm(Y - np.dot(U, V), 'fro'), 2)
        part5 = beta * pow(LA.norm(X - np.dot(U, M), 'fro'), 2)
        part3 = gamma * np.trace(np.dot(np.dot(U.T, Lx), U))
        part4 = 2 * theta * np.trace(np.dot(np.dot(temp_v.T, D), temp_v))
        objectives = part1 + part2 + part3 + part4 + part5

        obj.append(objectives)
        cver = abs((objectives - obji) / float(obji))
        obji = objectives
        iter = iter + 1
        if (iter > 2 and (cver < 1e-3 or iter == 1000)):
            break

    return temp_v
