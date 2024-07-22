""" Test the scatter plot on numerical data """

import os
import numpy as np
import matplotlib.pyplot as plt

from pyfd.decompositions import get_components_brute_force, get_components_adaptive
from pyfd.decompositions import get_interventional_from_anchored
from pyfd.shapley import shap_from_decomposition, permutation_shap
from pyfd.plots import attrib_scatter_plot
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

    # Model to explain
    def h(X):
        return alpha * X[:, 0] * X[:, 1] * (X[:, 2]>=0).astype(np.int64) +\
                beta * X[:, 3] * X[:, 4] * (X[:, 2]<0).astype(np.int64)

    return X, h

X, h = generate_problem(1000, 42, 0.2, 0.5, 1, 2)
features = Features(X, feature_names=[f"x{i}" for i in range(1, 6)], feature_types=["num"]*5)
features.summary()

###################################################################
#                No feature grouping nor regions                  #
###################################################################

# Anchored Additive decomposition
decomposition = get_components_brute_force(h, X, X)
shap_values = permutation_shap(h, X, X)
attrib_scatter_plot(decomposition, shap_values, foreground=X, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "0_scatter_anchored_add.png"), bbox_inches="tight")

# Full Anchored decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, tolerance=1e-5)
shap_values = shap_from_decomposition(decomposition)
attrib_scatter_plot(decomposition, shap_values, X, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "1_scatter_anchored_full.png"), bbox_inches="tight")

# Interventional Additive decomposition
decomposition = get_components_brute_force(h, X, X, anchored=False)
shap_values = permutation_shap(h, X, X)
attrib_scatter_plot(decomposition, shap_values, X, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "2_scatter_interv_add.png"), bbox_inches="tight")

# Full Interventional decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, tolerance=1e-5)
decomposition = get_interventional_from_anchored(decomposition)
shap_values = shap_from_decomposition(decomposition)
attrib_scatter_plot(decomposition, shap_values, X, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "3_scatter_interv_full.png"), bbox_inches="tight")

# Play with n_cols and idxs

# Top - 3
attrib_scatter_plot(decomposition, shap_values, X, idxs=3, 
                    features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "4_scatter_top_3.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(decomposition, shap_values, X, idxs=3,
                    features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "5_scatter_top_3_next_row.png"), bbox_inches="tight")

# Specific idxs
attrib_scatter_plot(decomposition, shap_values, X, idxs=[0, 2, 4],
                    features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "6_scatter_idxs_024.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(decomposition, shap_values, X, idxs=[0, 2 ,4], 
                    features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "7_scatter_idxs_024_next_row.png"), bbox_inches="tight")


###################################################################
#                            Regions                              #
###################################################################

regions = (X[:, 3] <= 0).astype(int)
regional_backgrounds = [[], []]
for r in range(2):
    regional_backgrounds[r] = X[regions == r]

# Regional Anchored Additive decomposition
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_brute_force(h, regional_backgrounds[r], 
                                                           regional_backgrounds[r])
    regional_shap[r] = permutation_shap(h, regional_backgrounds[r], 
                                        regional_backgrounds[r])
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "8_scatter_regional_anchored_add.png"), bbox_inches="tight")

# Grouped Full Anchored decomposition (should only show x3)
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_adaptive(h, regional_backgrounds[r],
                                                           tolerance=1e-5)
    regional_shap[r] = shap_from_decomposition(regional_decompositions[r])
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "9_scatter_regional_anchored_full.png"), bbox_inches="tight")


# Regional Interventional Additive decomposition
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_brute_force(h, regional_backgrounds[r], 
                                                           regional_backgrounds[r],
                                                           anchored=False)
    regional_shap[r] = permutation_shap(h, regional_backgrounds[r], 
                                        regional_backgrounds[r])
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "10_scatter_regional_interv_add.png"), bbox_inches="tight")

# Grouped Full Anchored decomposition (should only show x3)
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_adaptive(h, regional_backgrounds[r],
                                                           tolerance=1e-5)
    regional_decompositions[r] = get_interventional_from_anchored(regional_decompositions[r])
    regional_shap[r] = shap_from_decomposition(regional_decompositions[r])
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "11_scatter_regional_interv_full.png"), bbox_inches="tight")


# Top - 3
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, 
                    idxs=3, features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "12_scatter_region_top_3.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, 
                    idxs=3, features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "13_scatter_region_top_3_next_row.png"), bbox_inches="tight")

# Specific idxs
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds,
                    idxs=[0, 2, 4], features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "14_scatter_region_idxs_024.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds,
                    idxs=[0, 2, 4], features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "15_scatter_region_idxs_024_next_row.png"), bbox_inches="tight")


###################################################################
#                        Feature Grouping                         #
###################################################################

grouped_features = features.group([[0, 1]])
grouped_features.summary()

# Anchored Additive decomposition
decomposition = get_components_brute_force(h, X, X, Imap_inv=grouped_features.Imap_inv)
shap_values = permutation_shap(h, X, X, Imap_inv=grouped_features.Imap_inv, M=40)
attrib_scatter_plot(decomposition, shap_values, foreground=X,
                    features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "16_scatter_group_anchored_add.png"), bbox_inches="tight")

# Full Anchored decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, Imap_inv=grouped_features.Imap_inv, tolerance=1e-5)
shap_values = shap_from_decomposition(decomposition)
attrib_scatter_plot(decomposition, shap_values, X, 
                    features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "17_scatter_group_anchored_full.png"), bbox_inches="tight")

# Interventional Additive decomposition
decomposition = get_components_brute_force(h, X, X, Imap_inv=grouped_features.Imap_inv, 
                                                    anchored=False)
shap_values = permutation_shap(h, X, X, Imap_inv=grouped_features.Imap_inv, M=40)
attrib_scatter_plot(decomposition, shap_values, X, 
                    features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "18_scatter_group_interv_add.png"), bbox_inches="tight")

# Full Interventional decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, Imap_inv=grouped_features.Imap_inv, tolerance=1e-5)
decomposition = get_interventional_from_anchored(decomposition)
shap_values = shap_from_decomposition(decomposition)
attrib_scatter_plot(decomposition, shap_values, X, 
                    features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "19_scatter_group_interv_full.png"), bbox_inches="tight")

# Play with n_cols and idxs

# Top - 3
attrib_scatter_plot(decomposition, shap_values, X, idxs=2, 
                    features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "20_scatter_group_top_2.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(decomposition, shap_values, X, idxs=2,
                    features=grouped_features, n_cols=1)
plt.savefig(os.path.join("Images", "21_scatter_group_top_2_next_row.png"), bbox_inches="tight")

# Specific idxs
attrib_scatter_plot(decomposition, shap_values, X, idxs=[0, 2, 3],
                    features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "22_scatter_group_idxs_023.png"), bbox_inches="tight")

# Will not go to the next row because grouped features are ignored
attrib_scatter_plot(decomposition, shap_values, X, idxs=[0, 2 ,3], 
                    features=grouped_features, n_cols=2)
plt.savefig(os.path.join("Images", "23_scatter_group_idxs_023_next_row.png"), bbox_inches="tight")
