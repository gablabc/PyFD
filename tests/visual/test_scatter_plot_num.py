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
shap_values = permutation_shap(h, X, X, features)
attrib_scatter_plot(decomposition, shap_values, foreground=X, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "scatter_0_anchored_add.png"), bbox_inches="tight")

# Full Anchored decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, features, tolerance=1e-5)
shap_values = shap_from_decomposition(decomposition)
attrib_scatter_plot(decomposition, shap_values, X, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "scatter_1_anchored_full.png"), bbox_inches="tight")

# Interventional Additive decomposition
decomposition = get_components_brute_force(h, X, X, features, anchored=False)
shap_values = permutation_shap(h, X, X, features)
attrib_scatter_plot(decomposition, shap_values, X, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "scatter_2_interv_add.png"), bbox_inches="tight")

# Full Interventional decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, features, tolerance=1e-5)
decomposition = get_interventional_from_anchored(decomposition)
shap_values = shap_from_decomposition(decomposition)
attrib_scatter_plot(decomposition, shap_values, X, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "scatter_3_interv_full.png"), bbox_inches="tight")

# Play with n_cols and idxs

# Top - 3
attrib_scatter_plot(decomposition, shap_values, X, idxs=3, features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "scatter_4_top_3.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(decomposition, shap_values, X, idxs=3, features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "scatter_5_top_3_next_row.png"), bbox_inches="tight")

# Specific idxs
attrib_scatter_plot(decomposition, shap_values, X, idxs=[0, 2, 4], features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "scatter_6_idxs_024.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(decomposition, shap_values, X, idxs=[0, 2 ,4], features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "scatter_7_idxs_024_next_row.png"), bbox_inches="tight")


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
    regional_decompositions[r] = get_components_brute_force(h, regional_backgrounds[r], regional_backgrounds[r], features)
    regional_shap[r] = permutation_shap(h, regional_backgrounds[r], regional_backgrounds[r], features)
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "scatter_8_regional_anchored_add.png"), bbox_inches="tight")

# Grouped Full Anchored decomposition
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_adaptive(h, regional_backgrounds[r], features, tolerance=1e-5)
    regional_shap[r] = shap_from_decomposition(regional_decompositions[r])
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "scatter_9_regional_anchored_full.png"), bbox_inches="tight")


# Regional Interventional Additive decomposition
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_brute_force(h, regional_backgrounds[r], regional_backgrounds[r],
                                                           features, anchored=False)
    regional_shap[r] = permutation_shap(h, regional_backgrounds[r], regional_backgrounds[r], features)
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, 
                    features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "scatter_10_regional_interv_add.png"), bbox_inches="tight")

# Grouped Full Anchored decomposition 
regional_decompositions = [[], []]
regional_shap = [[], []]
for r in range(2):
    regional_decompositions[r] = get_components_adaptive(h, regional_backgrounds[r], features, tolerance=1e-5)
    regional_decompositions[r] = get_interventional_from_anchored(regional_decompositions[r])
    regional_shap[r] = shap_from_decomposition(regional_decompositions[r])
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, features=features, n_cols=5, figsize=(18, 4))
plt.savefig(os.path.join("Images", "scatter_11_regional_interv_full.png"), bbox_inches="tight")


# Top - 3
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds,
                    idxs=3, features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "scatter_12_region_top_3.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds, 
                    idxs=3, features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "scatter_13_region_top_3_next_row.png"), bbox_inches="tight")

# Specific idxs
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds,
                    idxs=[0, 2, 4], features=features, n_cols=5, figsize=(12, 4))
plt.savefig(os.path.join("Images", "scatter_14_region_idxs_024.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(regional_decompositions, regional_shap, regional_backgrounds,
                    idxs=[0, 2, 4], features=features, n_cols=2, figsize=(8, 8))
plt.savefig(os.path.join("Images", "scatter_15_region_idxs_024_next_row.png"), bbox_inches="tight")


###################################################################
#                        Feature Grouping                         #
###################################################################

grouped_features = features.group([[0, 1]])
grouped_features.summary()
# Here grouped features should be ignored (raising a warning) since we cannot plot
# their main effect as a line chart,

# Anchored Additive decomposition
decomposition = get_components_brute_force(h, X, X, grouped_features)
shap_values = permutation_shap(h, X, X, grouped_features, M=40)
attrib_scatter_plot(decomposition, shap_values, foreground=X, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "scatter_16_group_anchored_add.png"), bbox_inches="tight")

# Full Anchored decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, grouped_features, tolerance=1e-5)
shap_values = shap_from_decomposition(decomposition)
attrib_scatter_plot(decomposition, shap_values, X, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "scatter_17_group_anchored_full.png"), bbox_inches="tight")

# Interventional Additive decomposition
decomposition = get_components_brute_force(h, X, X, grouped_features, anchored=False)
shap_values = permutation_shap(h, X, X, grouped_features, M=40)
attrib_scatter_plot(decomposition, shap_values, X, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "scatter_18_group_interv_add.png"), bbox_inches="tight")

# Full Interventional decomposition (PDPs should ignore non-additive keys)
decomposition = get_components_adaptive(h, X, grouped_features, tolerance=1e-5)
decomposition = get_interventional_from_anchored(decomposition)
shap_values = shap_from_decomposition(decomposition)
attrib_scatter_plot(decomposition, shap_values, X, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "scatter_19_group_interv_full.png"), bbox_inches="tight")

# Play with n_cols and idxs

# Top - 3
attrib_scatter_plot(decomposition, shap_values, X, idxs=2, features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "scatter_20_group_top_2.png"), bbox_inches="tight")

# Go to the next row
attrib_scatter_plot(decomposition, shap_values, X, idxs=2, features=grouped_features, n_cols=1)
plt.savefig(os.path.join("Images", "scatter_21_group_top_2_next_row.png"), bbox_inches="tight")

# Specific idxs
attrib_scatter_plot(decomposition, shap_values, X, idxs=[0, 2, 3], features=grouped_features, n_cols=5)
plt.savefig(os.path.join("Images", "scatter_22_group_idxs_023.png"), bbox_inches="tight")

# Will not go to the next row because grouped features are ignored
attrib_scatter_plot(decomposition, shap_values, X, idxs=[0, 2 ,3], features=grouped_features, n_cols=2)
plt.savefig(os.path.join("Images", "scatter_23_group_idxs_023_next_row.png"), bbox_inches="tight")
