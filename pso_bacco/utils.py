#!/usr/bin/env python
# -*- coding: utf-8 -*-

def latinhypercube(n, samples, limits=None, spread=True, seed=None):
    import numpy as np 
    np.random.seed(seed)

    def _spread_lh(values, ndims, max_niter=10000):

        nrejected = 0
        total_dist = 0
        for x in range(ndims):
            for y in range(0, x):
                dist, ind = nearest_neighbour(values[:, [x, y]])
                total_dist += np.sum(dist)

        for i in range(max_niter):
            # We select two random points without replacement
            i1, i2 = np.random.choice(len(ind), 2, replace=False)

            # Let's swap values and accept it if it creates a more homogeneous distribution
            for dim in range(ndims):
                values[i1, dim], values[i2, dim] = values[i2, dim], values[i1, dim]
                new_total_dist = 0
                for x in range(ndims):
                    for y in range(0, x):
                        dist, ind = nearest_neighbour(values[:, [x, y]])
                        new_total_dist += np.sum(dist)

                if (new_total_dist < total_dist):
                    values[i1, dim], values[i2, dim] = values[i2, dim], values[i1, dim]
                    nrejected += 1
                    if nrejected == 2000:
                        return values
                else:
                    nrejected = 0
                    total_dist = new_total_dist

        return values 

    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a

    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = np.random.permutation(range(samples))
        H[:, j] = rdpoints[order, j]

    if spread is True:
        H = _spread_lh(H, n)

    if limits is not None:
        H = (H *(limits[:,1]-limits[:,0])+limits[:,0])

    return H


def set_rcParam(useTex=True):
    """Alternative styles for the plots
    
    :param useTex: Use Latex, defaults to True
    :type useTex: bool, optional
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    params = {"text.usetex": useTex,
              
              "axes.labelpad":              10,
              "axes.labelsize":             7,
              "axes.linewidth":             2,
              "axes.labelpad":              10,
              
              "xtick.labelsize":            33,
              "xtick.bottom":               True,
              "xtick.top":                  True,
              "xtick.direction":            'in',
              "xtick.minor.visible":        True,
              "xtick.minor.size":           6,
              "xtick.minor.width":          1,
              "xtick.minor.pad":            4,
              "xtick.major.size":           12,
              "xtick.major.width":          2,
              "xtick.major.pad":            3,

              "ytick.labelsize":            33,
              "ytick.left":                 True,
              "ytick.right":                True,
              "ytick.direction":            'in',
              "ytick.minor.visible":        True,
              "ytick.minor.size":           6,
              "ytick.minor.width":          1,
              "ytick.minor.pad":            4,
              "ytick.major.size":           12,
              "ytick.major.width":          2,
              "ytick.major.pad":            3,

              "figure.figsize":             "10, 10",
              "figure.dpi":                 80,
              "figure.subplot.left":        0.05,
              "figure.subplot.bottom":      0.05,
              "figure.subplot.right":       0.95,
              "figure.subplot.top":         0.95,

              "legend.numpoints":           1,
              "legend.frameon":             False,
              "legend.handletextpad":       0.3,

              "savefig.dpi":                80,

              "font.family":          'serif',

              "path.simplify":              True

              }

    # plt.rc("font", family = "serif")
    plt.rcParams.update(params)

