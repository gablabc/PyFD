# %% [markdown]
# # Additional Example of FDTrees
# In this tutorial, we will look at another toy example of FDTrees.
# %%
import numpy as np
import matplotlib.pyplot as plt
import shap
from shap.maskers import Independent
from sklearn.neural_network import MLPRegressor

from pyfd.features import Features
from pyfd.fd_trees import CoE_Tree, PDP_PFI_Tree, GADGET_PDP
from pyfd.decompositions import get_components_brute_force, get_PDP_PFI_importance
from pyfd.plots import setup_pyplot_font, bar, attrib_scatter_plot, plot_legend

setup_pyplot_font(30)

# %% [markdown]
# We first generate toy data following $x_i \sim U(-1, 1)$ for $i=0, 1, 2, 3$
# and $$y=
#   \begin{cases}
#         \sin(\pi x_2) & \text{if  } x_0\geq 0 \,\text{and}\, x_1 \geq 0\\
#         -2x_2^2 + x_3 & \text{otherwise},
#   \end{cases}$$
# These labels will be fit by a Multi-Layered Perceptron $h$ which is the model
# we are going to explain.
# %%
# Generate the data
np.random.seed(42)
d = 4
latex_feature_names = [r"$x_0$", r"$x_1$", r"$x_2$", r"$x_3$"]
X = np.random.uniform(-1, 1, size=(1000, d))
features = Features(X, latex_feature_names, ["num"]*d)
def h(X):
    y_hat = np.zeros((X.shape[0]))
    mask = (X[:, 0] > 0) & (X[:, 1] > 0)
    y_hat[mask] = np.sin(np.pi * X[mask, 2])
    y_hat[~mask] = -2 * X[~mask, 2]**2 + X[~mask, 3]
    return y_hat
y = h(X)
model = MLPRegressor(hidden_layer_sizes=(100, 50, 20, 10), max_iter=500).fit(X, y)
h = lambda x : model.predict(x)

# %% [markdown]
# The typical SHAP pipeline employs the whole dataset (or a random subsample) as the background
# distribution. 
# %%

# Run SHAP on whole dataset
background = X
masker = Independent(background, max_samples=background.shape[0])
explainer = shap.explainers.Exact(h, masker)
phis = explainer(background).values

# %% [markdown]
# The resulting Shapley values can be compared to an additive decomposition
# $$h_{i, \mathcal{B}}(x) = \mathbb{E}_{z\sim\mathcal{B}}[\,h(x_i, z_{-i})-h(z)]$$
# which is computed by PyFD as follows
# %%

# ANOVA Additive Decomposition
decomposition = get_components_brute_force(h, X, X)
attrib_scatter_plot(decomposition, phis, X, features)
plt.show()

# %% [markdown]
# In this plot, the lines are the additive decomposition while the
# points are the SHAP values. We observe strong disagreements between the additive 
# decomposition and SHAP. This disagreement also translates to the PDP/SHAP/PFI
# global feature importance.
# %%
# # Global feature importance
I_PDP, I_PFI  = get_PDP_PFI_importance(decomposition)
I_SHAP = (phis**2).mean(axis=0)
bar([I_PFI, I_SHAP, I_PDP], features.names())
plt.yticks(fontsize=35)
plt.xlabel("Feature Importance")
plt.show()

# %% [markdown]
# Again, we are going to fit a FDTree to increase the agreement between
# post-hoc explainers.
# %%

tree = CoE_Tree(features, max_depth=2, save_losses=True, branching_per_node=2)
# tree = PDP_PFI_Tree(features, max_depth=2, save_losses=True, branching_per_node=2)
# tree = GADGET_PDP(features, max_depth=2, save_losses=True, branching_per_node=2)
# Fit the tree
tree.fit(X, decomposition)
tree.print()


# Plot the objective values w.r.t the split candidates
plt.figure()
for i in range(3):
    splits = tree.root.splits[i]
    objectives = tree.root.objectives[i]
    plt.plot(splits, objectives, '-o', label=latex_feature_names[i])
plt.ylim(0, y.var())
plt.xlabel(r"Split threshold $\gamma$")
if isinstance(tree, CoE_Tree):
    plt.ylabel(r"$L_2$ Cost of Exclusion")
elif isinstance(tree, PDP_PFI_Tree):
    plt.ylabel("Loss PFI vs PDP")
elif isinstance(tree, GADGET_PDP):
    plt.ylabel("Loss ICE vs PDP")
plt.legend()
plt.show()

# %% [markdown]
# Once a FDTree is fitted, it can be used to partition
# the data samples into groups. The method `.predict()` 
# returns the group index for each datum. Moreover, the `.rules()`
# method returns an interpretable representation of the regions.
# %%
groups = tree.predict(X)
rules = tree.rules(use_latex=False)
print(rules)

# We rerun .predict() to return latex formated rules
rules = tree.rules(use_latex=True)

# %% [markdown]
# Given these regions, instead of using the whole dataset
# as background, we can iterate over all regions and
# only study the samples that land in said region.
# Regional Feature Importance can be computed by passing
# `groups` to `get_PDP_PFI_importance`.
# %%
# Global Feature Importance
I_PDP, I_PFI  = get_PDP_PFI_importance(decomposition, groups=groups)

# %% [markdown]
# Shapley values must be computed by calling SHAP various times.
# %%

# Rerun SHAP and PDP
phis_list = [0] * tree.n_groups
for group_idx in range(tree.n_groups):
    print(f"### Region {group_idx} ###")
    idx_select = (groups == group_idx)
    background = X[idx_select]

    # SHAP
    mu = h(background).mean()
    masker = Independent(background, max_samples=background.shape[0])
    explainer = shap.explainers.Exact(h, masker)
    phis_list[group_idx] = explainer(background).values

    I_SHAP = (phis_list[group_idx]**2).mean(axis=0)
    bar([I_PFI[group_idx], I_SHAP, I_PDP[group_idx]], features.names())
    # plt.gca().invert_yaxis()
    plt.yticks(fontsize=35)
    plt.xlabel("Feature Importance")
    plt.show()

# %% [markdown]
# Finally, we recompare the local feature attributions
# %%

attrib_scatter_plot(decomposition, phis_list, X, features, groups=groups)
plot_legend(rules, ncol=3)
plt.show()

# %% [markdown]
# As a result of explaining the model regionally, we have reduced the 
# disagreement between the post-hoc explainers.
# %%