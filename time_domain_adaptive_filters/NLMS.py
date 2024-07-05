""" Normalized Least Mean Squares Filter """

import numpy as np


def nlms(x, d, N=4, mu=0.1):
    nIters = min(len(x), len(d)) - N
    u = np.zeros(N)
    w = np.zeros(N)
    e = np.zeros(nIters)
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        e_n = d[n] - np.dot(u, w)
        w = w + mu * e_n * u / (np.dot(u, u) + 1e-3)
        e[n] = e_n
    return e
