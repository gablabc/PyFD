""" Detect extrapolation when explaining models """

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import spearmanr

from .utils import check_Imap_inv, ravel


def cluster_features_correlations(X, feature_names, threshold=0.2, plot_dendrogram=True):
    correl, _ = spearmanr(X)
    d = X.shape[1]
    # Hiearchical clustering of features based on correlation
    # High positive correlation -> small distance
    Z = linkage(1-correl[np.triu_indices(d, k=1)], 'single')
    if plot_dendrogram:
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z, orientation='right', labels=feature_names, 
                        above_threshold_color='k', color_threshold=threshold,
                        leaf_font_size=20)
        plt.plot(threshold*np.ones(2), [0, 200], 'k--')
        plt.xlabel("1-Spearman")
    
    cluster_idx = fcluster(Z, t=threshold, criterion="distance")
    clusters = []
    for idx, count in zip(*np.unique(cluster_idx, return_counts=True)):
        if count > 1:
            clusters.append(list(np.where(cluster_idx==idx)[0]))
    return clusters


def sample_synthetic_points(foreground, background, groups_method=None, Imap_inv=None, max_card=None, n_samples=1000, seed=42):
    
    # Setup
    np.random.seed(seed)
    d = foreground.shape[1]
    Nf = foreground.shape[0]
    Nb = background.shape[0]

    # Setup Imap_inv
    Imap_inv, D, _ = check_Imap_inv(Imap_inv, d)

    # Assign instances to groups
    if groups_method is None:
        foreground_groups = np.zeros(Nf)
        background_groups = np.zeros(Nb)
    else:
        foreground_groups = groups_method(foreground)
        background_groups = groups_method(background)

    # Sample proportional to each group
    groups, group_ratios = np.unique(np.append(foreground_groups, background_groups), return_counts=True)
    group_ratios = group_ratios / (Nf + Nb)

    # Sample low cardinality feature groups with higher probability
    combinatorics = np.ones(D)
    combinatorics[0] = D
    for i in range(1, D):
        combinatorics[i] = (D - i) / (i + 1) * combinatorics[i-1]
    U_probs = 2.0 ** (-np.arange(1, D+1))# * combinatorics
    if max_card is not None:
        U_probs = U_probs[:max_card]
    U_probs /= U_probs.sum()

    # Iterate over all groups
    samples = []
    for group, group_ratio in zip(groups, group_ratios):
        n_samples_group = int(n_samples * group_ratio)
        foreground_subset = foreground[foreground_groups == group]
        background_subset = background[background_groups == group]

        # Samples data instances
        index_f = np.random.choice(range(len(foreground_subset)), n_samples_group, replace=True)
        index_b = np.random.choice(range(len(background_subset)), n_samples_group, replace=True)

        # Sample subsets sizes
        U_size = np.random.choice(range(1, D+1), n_samples_group, replace=True, p=U_probs)

        # Generate synthetic data
        for i in range(n_samples_group):
            # Features to switch
            U = np.random.permutation(range(D))[:U_size[i]]
            U = ravel([Imap_inv[j] for j in U])
            # Sample from foreground and background
            synthetic_pts = background_subset[index_b[i]]
            synthetic_pts[U] = foreground_subset[index_f[i], U]
            samples.append(synthetic_pts)

    return np.vstack(samples)
