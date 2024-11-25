""" Test the partial dependence plot on numerical data """

import os
import numpy as np
import matplotlib.pyplot as plt

from pyfd.decompositions import get_components_brute_force, get_components_adaptive
from pyfd.decompositions import get_interventional_from_anchored, get_regional_decompositions
from pyfd.plots import partial_dependence_plot
from pyfd.features import Features

def generate_problem(N, seed, rho_12, rho_45, alpha, beta):
    # Generate input
    np.random.seed(seed)
    cov = np.eye(5)
    cov[0, 1] = rho_12
    cov[1, 0] = rho_12
    cov[3, 4] = rho_45
    cov[4, 3] = rho_45
    X = np.random.multivariate_normal(np.zeros(5), cov=cov, size=(N,))
    features = Features(X, [f"x{i}" for i in range(1, 6)], ["num"]*5)
    # Model to explain
    def h(X):
        return alpha * X[:, 0] * X[:, 1] * (X[:, 2]>=0).astype(np.int64) +\
                beta * X[:, 3] * X[:, 4] * (X[:, 2]<0).astype(np.int64)

    return X, h, features

X, h, features = generate_problem(1000, 42, 0.2, 0.5, 1, 2)
features.summary()

###################################################################
#                No feature grouping nor regions                  #
###################################################################

# Anchored Additive decomposition
decomposition = get_components_brute_force(h, X, X, features)
partial_dependence_plot(decomposition, foreground=X, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "pdp_0_anchored_add.png"), bbox_inches="tight")

# Full Anchored decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, features, tolerance=1e-5)
partial_dependence_plot(decomposition, X, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "pdp_1_anchored_full.png"), bbox_inches="tight")

# Interventional Additive decomposition
decomposition = get_components_brute_force(h, X, X, features, anchored=False)
partial_dependence_plot(decomposition, X, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "pdp_2_interv_add.png"), bbox_inches="tight")

# Full Interventional decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, features, tolerance=1e-5)
decomposition = get_interventional_from_anchored(decomposition)
partial_dependence_plot(decomposition, X, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "pdp_3_interv_full.png"), bbox_inches="tight")

# Play with n_cols and idxs

# Top - 3
partial_dependence_plot(decomposition, X, idxs=3, features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "pdp_4_top_3.png"), bbox_inches="tight")

# Go to the next row
partial_dependence_plot(decomposition, X, idxs=3, features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "pdp_5_top_3_next_row.png"), bbox_inches="tight")

# Specific idxs
partial_dependence_plot(decomposition, X, idxs=[0, 2, 4], features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "pdp_6_idxs_024.png"), bbox_inches="tight")

# Go to the next row
partial_dependence_plot(decomposition, X, idxs=[0, 2 ,4], features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "pdp_7_idxs_024_next_row.png"), bbox_inches="tight")


####################################################################
##                            Regions                              #
####################################################################

regions = (X[:, 3] <= 0).astype(int)
regional_backgrounds = [[], []]
for r in range(2):
    regional_backgrounds[r] = X[regions == r]

# Regional Anchored Additive decomposition
regional_decompositions = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_brute_force(h, regional_backgrounds[r], regional_backgrounds[r], features)
partial_dependence_plot(regional_decompositions, regional_backgrounds, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "pdp_8_regional_anchored_add.png"), bbox_inches="tight")

# Grouped Full Anchored decomposition
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_adaptive(h, regional_backgrounds[r], features, tolerance=1e-5)
partial_dependence_plot(regional_decompositions, regional_backgrounds, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "pdp_9_regional_anchored_full.png"), bbox_inches="tight")


# Regional Interventional Additive decomposition
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_brute_force(h, regional_backgrounds[r], 
                                                           regional_backgrounds[r],
                                                           features,
                                                           anchored=False)
partial_dependence_plot(regional_decompositions, regional_backgrounds, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "pdp_10_regional_interv_add.png"), bbox_inches="tight")

# Grouped Full Anchored decomposition 
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_adaptive(h, regional_backgrounds[r], features, tolerance=1e-5)
    regional_decompositions[r] = get_interventional_from_anchored(regional_decompositions[r])
partial_dependence_plot(regional_decompositions, regional_backgrounds, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "pdp_11_regional_interv_full.png"), bbox_inches="tight")


# Top - 3
partial_dependence_plot(regional_decompositions, regional_backgrounds, idxs=3, features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "pdp_12_region_top_3.png"), bbox_inches="tight")

# Go to the next row
partial_dependence_plot(regional_decompositions, regional_backgrounds, idxs=3, features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "pdp_13_region_top_3_next_row.png"), bbox_inches="tight")

# Specific idxs
partial_dependence_plot(regional_decompositions, regional_backgrounds, idxs=[0, 2, 4], features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "pdp_14_region_idxs_024.png"), bbox_inches="tight")

# Go to the next row
partial_dependence_plot(regional_decompositions, regional_backgrounds, idxs=[0, 2, 4], features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "pdp_15_region_idxs_024_next_row.png"), bbox_inches="tight")


####################################################################
##                        Feature Grouping                         #
####################################################################

grouped_features = features.group([[0, 1]])
grouped_features.summary()

# Anchored Additive decomposition
decomposition = get_components_brute_force(h, X, X, grouped_features)
partial_dependence_plot(decomposition, foreground=X, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "pdp_16_group_anchored_add.png"), bbox_inches="tight")

# Full Anchored decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, grouped_features, tolerance=1e-5)
partial_dependence_plot(decomposition, X, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "pdp_17_group_anchored_full.png"), bbox_inches="tight")

# Interventional Additive decomposition
decomposition = get_components_brute_force(h, X, X, grouped_features, anchored=False)
partial_dependence_plot(decomposition, X, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "pdp_18_group_interv_add.png"), bbox_inches="tight")

# Full Interventional decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, grouped_features, tolerance=1e-5)
decomposition = get_interventional_from_anchored(decomposition)
partial_dependence_plot(decomposition, X, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "pdp_19_group_interv_full.png"), bbox_inches="tight")

# Play with n_cols and idxs

# Top - 3
partial_dependence_plot(decomposition, X, idxs=2, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "pdp_20_group_top_2.png"), bbox_inches="tight")

# Go to the next row
partial_dependence_plot(decomposition, X, idxs=2, features=grouped_features, n_cols=1)
plt.savefig(os.path.join("Images", "pdp_21_group_top_2_next_row.png"), bbox_inches="tight")

# Specific idxs
partial_dependence_plot(decomposition, X, idxs=[0, 2, 3], features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "pdp_22_group_idxs_023.png"), bbox_inches="tight")

# Will not go to the next row because grouped features are ignored
partial_dependence_plot(decomposition, X, idxs=[0, 2 ,3], features=grouped_features, n_cols=2)
plt.savefig(os.path.join("Images", "pdp_23_group_idxs_023_next_row.png"), bbox_inches="tight")
