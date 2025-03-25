'''Kernelized General Fuzzy c-Means Clustering

Paper Source: Gupta A., Das S., 'On the Unification of k-Harmonic Means and
Fuzzy c-Means Clustering Problems under Kernelization', in the 2017 Ninth
International Conference on Advances in Pattern Recognition (ICAPR 2017),
pp. 386-391, 2017.
'''

# Author: Avisek Gupta

import numpy as np
from scipy.spatial.distance import cdist

def kernel_gfcm(X, n_clusters, sigma=1, m=2, p=2, max_iter=300, n_init=30):
    """Kernelized General Fuzzy c-Means Clustering"""
    tol = 1e-16
    min_cost = np.float64('inf')  # Changed from 1000000
    eps = np.finfo(np.float64).eps  # Define eps once
    
    for _ in range(n_init):
        centers = X[np.random.choice(
            X.shape[0], size=n_clusters, replace=False
        )]
        for v_iter in range(max_iter):
            # Compute kernel similarities
            K = np.exp(
                -cdist(centers, X, metric='sqeuclidean') / (2 * (sigma ** 2))
            )
            K_dist = np.fmax(1 - K, eps)
            
            # Update memberships
            U = np.fmax(
                K_dist ** (-p / (2 * (m - 1))), eps
            )
            U = U / U.sum(axis=0)
            
            # Update centers
            old_centers = np.array(centers)
            expr_part = np.fmax(
                (U ** m) * (K_dist ** ((p - 2) / 2)) * K,
                eps
            )
            centers = expr_part.dot(X) / expr_part.sum(axis=1)[:, None]

            if ((centers - old_centers) ** 2).sum() < tol:
                break

        # Compute cost
        cost = ((U ** m) * (K_dist ** (p / 2))).sum()
        if cost < min_cost:
            min_cost = cost
            mincost_centers = np.array(centers)
            mincost_mem = U.argmax(axis=0)
    
    return mincost_centers, mincost_mem, min_cost