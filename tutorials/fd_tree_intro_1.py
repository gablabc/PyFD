# %% [markdown]
# # Motivation Example for FDTrees
# In this tutorial, we will look at the toy example which served as the motivation for FDTrees.
# %%
import numpy as np
import matplotlib.pyplot as plt
import shap
from shap.maskers import Independent

from pyfd.features import Features
from pyfd.fd_trees import CoE_Tree, PDP_PFI_Tree, GADGET_PDP
from pyfd.decompositions import get_components_brute_force, get_PDP_PFI_importance
from pyfd.plots import setup_pyplot_font, bar, attrib_scatter_plot, plot_legend

setup_pyplot_font(15)

# %% [markdown]
# We first generate toy data following $x_i \sim U(-1, 1)$ for $i=0, 1, 2, 3, 4$
# and $$h(\bm{x})=
#   \begin{cases}
#         x_0 & \text{if  } x_1\geq 0\\
#         x_2 & \text{otherwise},
#   \end{cases}$$
# %%
np.random.seed(42)
d = 5
latex_feature_names = [r"$x_0$", r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
X = np.random.uniform(-1, 1, size=(1000, d))
features = Features(X, latex_feature_names, ["num"]*d)
def h(X):
    y_hat = np.zeros((X.shape[0]))
    mask = (X[:, 1] > 0)
    y_hat[mask] = X[mask, 0]
    y_hat[~mask] = X[~mask, 2]
    return y_hat
y = h(X)

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

decomposition = get_components_brute_force(h, X, X)
attrib_scatter_plot(decomposition, phis, X, features, figsize=(16, 4))
plt.show()

# %% [markdown]
# In this plot, the lines are the additive decomposition while the
# points are the SHAP values. We observe strong disagreements between the additive 
# decomposition and SHAP. This disagreement also translates to the PDP/SHAP/PFI
# global feature importance.
# %%
# Global feature importance
I_PDP, I_PFI  = get_PDP_PFI_importance(decomposition)
I_SHAP = (phis**2).mean(axis=0)
bar([I_PFI, I_SHAP, I_PDP], features.names())
plt.yticks(fontsize=35)
plt.xlabel("Feature Importance")
plt.show()

# %% [markdown]
# In this example, the importance of $x_1$ is uncertain because different explanation techniques give
# it vastly different importance. But what is the source of this disagreement? 
# The answer is **feature interactions**. Indeed, without interactions all explainers would agree. 
# Yet because $x_1$ interatcs with $x_0$ and $x_2$, the explainers cannot agree on its importance.
#
# How do we reduce feature interactions? Well, $x_1$ interacts
# with $x_0$ and $x_2$ because the effect of varying $x_0$ and $x_2$
# depends on whether or not $x_1$ is positive. But, if we only 
# focus on instances for which $x_1$ is positive (or negative),
# then this interactions ceases to exist. The idea behind FDTrees is therefore
# to partition the input space into regions where interactions are reduced.
# %%

# Fit the tree
tree = CoE_Tree(features, max_depth=1, save_losses=True)
# tree = PDP_PFI_Tree(features, max_depth=1, save_losses=True)
# tree = GADGET_PDP(features, max_depth=1, save_losses=True)
tree.fit(X, decomposition)
# Print results
tree.print(verbose=True)

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
# only study the samples that land in them.
# Regional Feature Importance can also be computed by passing
# `groups` to `get_PDP_PFI_importance`.
# %%
# Global Feature Importance
I_PDP, I_PFI  = get_PDP_PFI_importance(decomposition, groups=groups)

# %% [markdown]
# Shapley values must be computed by calling SHAP various times.
# %%
from pyfd.plots import COLORS

fig, axes = plt.subplots(1, tree.n_groups, figsize=(8, 4))
# Rerun SHAP and recompute global importance regionally
phis_list = [0] * tree.n_groups
for i in range(tree.n_groups):
    idx_select = (groups == i)
    background = X[idx_select]

    # SHAP
    masker = Independent(background, max_samples=background.shape[0])
    explainer = shap.explainers.Exact(h, masker)
    phis_list[i] = explainer(background).values

    I_SHAP = (phis_list[i]**2).mean(axis=0)
    bar([I_PFI[i], I_SHAP, I_PDP[i]], features.names(), ax=axes[i], color=COLORS[i])
    axes[i].set_xlim(0, np.max(I_PFI)+0.02)
    axes[i].set_xlabel("Feature Importance")
    axes[i].set_title(rules[i])
plt.show()

# %% [markdown]
# Finally, we recompare the local feature attributions
# %%

attrib_scatter_plot(decomposition, phis_list, X, features, 
                    groups=groups, figsize=(16, 4))
plot_legend(rules, ncol=2)
plt.show()

# %% [markdown]
# As a result of explaining the model regionally, we have reduced the 
# disagreement between the post-hoc explainers.
# %%