from functools import cache
import sys

import numpy as np
from scipy.stats import kendalltau, weightedtau

from .utils import kendall_tau, weighted_kendall_tau, grouped_weighted_kendall_tau


def check_similar(msg, v1, v2, tol=-6):
    eps = 10 ** tol
    if abs(v1 - v2) > eps:
        print(f"\n{msg}: {v1} != {v2} (within tol=1e{tol})")
        sys.exit()


weighter = lambda r: 1./(1+r)


@cache
def total_weight(n):
    ranks = np.arange(n)
    w = 0
    for i in range(n):
        for j in range(i+1, n):
            w += 2*(weighter(ranks[i]) + weighter(ranks[j]))
    return w


for _ in range(100):
    print('.', end='', flush=True)
    n = np.random.randint(1, 100)
    x = np.random.rand(n)
    y = np.random.rand(n)
    z = np.random.rand(n)
    w = np.random.rand(n)
    x2 = np.concatenate([x, x, w]) # Ties, but not joint ties
    y2 = np.concatenate([y, z, y])
    x3 = np.concatenate([x, x, x, w]) # Joint and non-joint ties
    y3 = np.concatenate([y, y, z, y])
    x4 = np.concatenate([x, x, w, w]) # Only joint ties
    y4 = np.concatenate([y, y, z, z])

    # Vanilla kendall tau

    k1 = kendall_tau(x, y)
    k2 = kendall_tau(y, x)
    kref = kendalltau(x, y).statistic
    check_similar("kendall_tau x,y vs. y,x", k1, k2)
    check_similar("kendall_tau x,y vs. ref", k1, kref)

    k1 = kendall_tau(x2, y2)
    k2 = kendall_tau(y2, x2)
    kref = kendalltau(x2, y2).statistic
    check_similar("kendall_tau x2,y2 vs. y2,x2", k1, k2)
    check_similar("kendall_tau x2,y2 vs. ref", k1, kref)

    k1 = kendall_tau(x3, y3)
    k2 = kendall_tau(y3, x3)
    kref = kendalltau(x3, y3).statistic
    check_similar("kendall_tau x3,y3 vs. y3,x3", k1, k2)
    check_similar("kendall_tau x3,y3 vs. ref", k1, kref)

    k1 = kendall_tau(x4, y4)
    k2 = kendall_tau(y4, x4)
    kref = kendalltau(x4, y4).statistic
    check_similar("kendall_tau x4,y4 vs. y4,x4", k1, k2)
    check_similar("kendall_tau x4,y4 vs. ref", k1, kref)

    # Weighted tau equivalence

    k1 = weighted_kendall_tau(x, y, weighter=lambda r: 1)
    k2 = kendall_tau(x, y)
    check_similar("weighted_kendall_tau (constant weighter) x,y vs. kendall_tau", k1, k2)

    k1 = weighted_kendall_tau(x2, y2, weighter=lambda r: 1)
    k2 = kendall_tau(x2, y2)
    check_similar("weighted_kendall_tau (constant weighter) x2,y2 vs. kendall_tau", k1, k2)

    k1 = weighted_kendall_tau(x3, y3, weighter=lambda r: 1)
    k2 = kendall_tau(x3, y3)
    check_similar("weighted_kendall_tau (constant weighter) x3,y3 vs. kendall_tau", k1, k2)

    k1 = weighted_kendall_tau(x4, y4, weighter=lambda r: 1)
    k2 = kendall_tau(x4, y4)
    check_similar("weighted_kendall_tau (constant weighter) x4,y4 vs. kendall_tau", k1, k2)

    # Weighted tau with reference

    k1 = weighted_kendall_tau(x, y)
    k2 = weighted_kendall_tau(y, x)
    kref = weightedtau(x, y).statistic
    check_similar("weighted_kendall_tau x,y vs. y,x", k1, k2)
    check_similar("weighted_kendall_tau x,y vs. ref", k1, kref, tol=-6)

    k1 = weighted_kendall_tau(x2, y2)
    k2 = weighted_kendall_tau(y2, x2)
    kref = weightedtau(x2, y2).statistic
    check_similar("weighted_kendall_tau x2,y2 vs. y2,x2", k1, k2)
    check_similar("weighted_kendall_tau x2,y2 vs. ref", k1, kref, tol=-3)

    k1 = weighted_kendall_tau(x3, y3)
    k2 = weighted_kendall_tau(y3, x3)
    kref = weightedtau(x3, y3).statistic
    check_similar("weighted_kendall_tau x3,y3 vs. y2,x2", k1, k2)
    check_similar("weighted_kendall_tau x3,y3 vs. ref", k1, kref, tol=-3)

    k1 = weighted_kendall_tau(x4, y4)
    k2 = weighted_kendall_tau(y4, x4)
    kref = weightedtau(x4, y4).statistic
    check_similar("weighted_kendall_tau x4,y4 vs. y4,x4", k1, k2)
    check_similar("weighted_kendall_tau x4,y4 vs. ref", k1, kref, tol=-3)

    # Grouped weighted tau equivalence

    k1 = weighted_kendall_tau(x, y)
    k2 = grouped_weighted_kendall_tau(x, y, [1] * len(x))
    check_similar("grouped_weighted_kendall_tau (single group) x,y vs. weighted_kendall_tau", k1, k2)

    k1 = weighted_kendall_tau(x2, y2)
    k2 = grouped_weighted_kendall_tau(x2, y2, [1] * len(x2))
    check_similar("grouped_weighted_kendall_tau (single group) x2,y2 vs. weighted_kendall_tau", k1, k2)

    k1 = weighted_kendall_tau(x3, y3)
    k2 = grouped_weighted_kendall_tau(x3, y3, [1] * len(x3))
    check_similar("grouped_weighted_kendall_tau (single group) x3,y3 vs. weighted_kendall_tau", k1, k2)

    k1 = weighted_kendall_tau(x4, y4)
    k2 = grouped_weighted_kendall_tau(x4, y4, [1] * len(x4))
    check_similar("grouped_weighted_kendall_tau (single group) x4,y4 vs. weighted_kendall_tau", k1, k2)

    # Predictable formula for grouped weighted tau
    groups = np.random.randint(2, 10)
    xg = []
    yg = []
    gg = []
    taus = []
    ws = []
    for g in range(groups):
        n = np.random.randint(1, 100)
        x = np.unique(np.random.rand(2*n))[:n]
        y = np.unique(np.random.rand(2*n))[:n]
        xg.append(x)
        yg.append(y)
        gg.append(np.array([g] * n))
        w = total_weight(n)
        ws.append(w)
        tau = weighted_kendall_tau(x, y)
        taus.append(tau * w)
    x = np.concatenate(xg)
    y = np.concatenate(yg)
    g = np.concatenate(gg)
    k1 = grouped_weighted_kendall_tau(x, y, g)
    k2 = sum(taus) / sum(ws)
    check_similar("grouped_weighted_kendall_tau vs. expected value", k1, k2, tol=-4)

