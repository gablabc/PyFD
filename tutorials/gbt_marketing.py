# %% [markdown]
# # Second Introduction to the PyFD package
# This second tutorial will again discuss the general functionalities of the `PyFD` package.
# In-depth explanations of the API are left for subsequent tutorials. In this notebook, we will decompose a complex 
# GBT model trained on the [Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) 
# datset, which aims at predicting successes of phone calls during a Bank's marketing campaign.
# Let us see how to interpret the complex model in spite of the presence of strong feature interactions.
# %% 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score
from joblib import dump, load

# Local imports
from pyfd.data import get_data_marketing
from pyfd.plots import setup_pyplot_font

setup_pyplot_font(8)

# %% [markdown]
# ## Fitting model
# We start by loading the data and a `Features` object which stores
# information about the various features. Note that the data **X** must **always**
# be a numerical numpy array. Categorical features are assumed to have been ordinally encoded.
# %%

# `use_target_encoder=True` will ordinally encode categorical variables in terms of average target value
X, y, features = get_data_marketing(use_target_encoder=True)
print(features.summary())

# %% [markdown]
# ## Fitting model
# Having fetched the data, we fit a `GradientBoostingClassifier`.
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42,
                                                    stratify=y)
model = GradientBoostingClassifier(random_state=0)

# %%
grid = {"learning_rate": np.logspace(-2, 0, 10),
        "max_depth" : [3, 4, 5, 6, 7],
        "n_estimators" : [50, 100, 150, 200],
        'min_samples_leaf' : [1, 20, 40, 60, 80, 100]}

search = RandomizedSearchCV(
    model,
    cv=KFold(),
    scoring='f1',
    param_distributions=grid,
    verbose=2,
    n_iter=20,
    random_state=42
)
search.fit(X_train, y_train)

# %%
# Recover the optimal CV model
model = search.best_estimator_
res = search.cv_results_
cv_perf = np.nan_to_num(res['mean_test_score'], nan=1e10)
best_idx = np.argmax(cv_perf)

# %%

plt.figure()
plt.scatter(res['param_learning_rate'], cv_perf, c='b', alpha=0.75)
plt.plot(res['param_learning_rate'][best_idx], cv_perf[best_idx], 'r*', markersize=10, markeredgecolor='k')
plt.xlabel("Learning Rate")
plt.ylabel("Cross-Validated F1 Score")
plt.xscale('log')
# plt.ylim(97, 120)
plt.show()

# %%

# Pickle the model
dump(model, os.path.join("models", "marketing_gbt.joblib"))

# %%
# Load the model
model = load(os.path.join("models", "marketing_gbt.joblib"))
print(f1_score(model.predict(X_test), y_test))
# We are going to explain the logits not the 0/1 prediction
test_preds = model.decision_function(X_test)

# %%[markdown]
# ## Additive Decomposition
# This models is not incredible but it still manages to make okay predictions
# of call successes. As in the previous tutorial, we must fix a background dataset 
# distribution $\mathcal{B}$ and express $h$ as the following sum
# $$h(x) = \sum_{u\subseteq [d]}h_{u,\mathcal{B}}(x)$$
# where $h_{u,\mathcal{B}}$ only depends on the subvector $x_u$.
# Let's visualize the additive terms $h_{i,\mathcal{B}}$ of the decomposition
# which only depend on feature $x_i$.
# %%
from pyfd.decompositions import get_components_tree, get_CoE
from pyfd.plots import partial_dependence_plot

# Reference data
background = X_train[:5000]
# We evaluate the decomp on the whole test set
foreground = X_test
# Compute the functional decomposition
decomp = get_components_tree(model, foreground, background, algorithm="leaf")
# Plot the additive terms
partial_dependence_plot(decomp, foreground, background, features,
                        plot_hist=True, n_cols=4, figsize=(10, 12))
plt.show()

# %% [markdown]
# This plot already gives us some insight on how individual features
# impact the model. Yet, by only considering the additive terms from the
# decomposition, we are indirectly assuming that $h$ is additive. This is
# not true for Gradient Boosted Trees. To measure how **far** we are from
# additivity, we can compute the additive decomposition
# $$ h_{\text{add}}(x) = h_{\emptyset,\mathcal{B}} + \sum_{i=1}^d h_{i,\mathcal{B}}(x_i)$$
# which amounts to summing the curves plotted previously.
# This additive decomposition would faithfully describe the original model 
# $h(x) \approx h_{\text{add}}(x)$ if there were **no**
# strong feature interactions. However, for this model, the error
# $$\mathbb{E}[(h(x) - h_{\text{add}}(x))^2] / \mathbb{V}[h(x)]$$ 
# is large.
# %%
print(f"CoE : {get_CoE(decomp, anchored=False, foreground_preds=test_preds):.2f} %")

# %% [markdown]
# 60% of the variability in the model cannot be
# explained by its additive approximation and so the curves
# shown previously are not representative of the true model
# behavior. `PyFD` will help us deal with this issue. First, we must
# identify which features interact.
# %%
from pyfd.shapley import taylor_treeshap
from pyfd.plots import interactions_heatmap

# This implementation uses the `recurse` algorithm which needs
# a smaller background
background = X_train[:800]
Phis = taylor_treeshap(model, background, background)
print(Phis.shape)

# %% [markdown]
# Given the Shapley-Taylor indices, we can compute a heatmap of the
# pairwise interaction strenght among the features
# %%

interactions_heatmap(Phis, features.print_names())
plt.show()

# %% [markdown]
# The strongest interactions are between `month-day` and `month-contact`.
# The first interaction implies that we cannot decouple the effect of `day`
# from the effect of `month`. Since these two variables are semantically similar
# (they both clarify the time of the year), we can **group** them
# in a single meta-feature.
# %%
# Grouping day and month
grouped_features = features.group([[5, 6]])
print(grouped_features.summary())

# %% [markdown]
# Note that we have a new feature `day:month` representing the day and month
# for a specific datum. The column `I^-1({i})` in the summary table represents
# the pre-image of each feature i.e. what columns of *X* map to each feature. Here
# we have `I^-1({i})=[5, 6]` because columns 5 and 6 both encode the feature `day:month`.
# The `Imap_inv` attribute must be passed to the the decomposition algorithm to
# decompose the model $h$ while treating columns 5 and 6 as a single feature.
# %%
print(grouped_features.Imap_inv)

background = X_train[:5000]
foreground = X_test
decomp_grouped = get_components_tree(model, foreground, background, algorithm="leaf", 
                                    Imap_inv=grouped_features.Imap_inv)
print(f"CoE : {get_CoE(decomp_grouped, anchored=False, foreground_preds=test_preds):.2f} %")

# %% [markdown]
# We have halfed the CoE by considering month-day as a single feature. 
# Now, the additive contribution $h_{i,\mathcal{B}}(x)$ of this meta feature must
# be visualized as a 2D plot.
# %%
from matplotlib.colors import TwoSlopeNorm
phi = decomp_grouped[(14,)]

plt.figure()
plt.title("Additive effect of month-day")
plt.scatter(foreground[:, 5], foreground[:, 6], c=phi, 
            norm=TwoSlopeNorm(vmin=phi.min(), vcenter=0, vmax=phi.max()),
            cmap='seismic')
plt.xlabel("day")
plt.ylabel("month")
plt.colorbar()
plt.show()

# %% [markdown]
# The LoA is still 30% and to we must find a way to reduce it even more.
# We could fit a FDTree on the new decomposition to obtain disjoint regions
# with reduced interactions.
# %%
from pyfd.fd_trees import CoE_Tree
from pyfd.decompositions import get_PDP_PFI_importance
from pyfd.plots import bar, COLORS, plot_legend

# Compute an anchored decomposition requires smaller backgrounds
# FD-Trees require background=foreground
background = X_train[:1000]
# No grouping
decomp_sym = get_components_tree(model, background, background, 
                                        anchored=True)
# Grouping
decomp_grouped_sym = get_components_tree(model, background, background, 
                                        Imap_inv=grouped_features.Imap_inv, 
                                        anchored=True)

# %%
# Fit a depth-1 tree
tree = CoE_Tree(max_depth=1, features=features)
tree.fit(background, decomp_grouped_sym)
tree.print(verbose=True)
groups = tree.predict(background).astype(bool)
rules = tree.rules(use_latex=True)

# %% [markdown]
# The split is made along the feature `contact` that interacts 
# with `month`. Since the LoA has been significantly reduced
# (it went from 60% to 10%) we have a higher confidence that
# our global feature importance and local feature attributions are more
# faithful to the model.
# %%

# Using the whole background without feature grouping
I_PDP, I_PFI = get_PDP_PFI_importance(decomp_sym)
bar([np.sqrt(I_PFI), np.sqrt(I_PDP)], features.print_names())
plt.title("Full background without grouping")
plt.show()

# %%

# Using the whole background with feature grouping
I_PDP, I_PFI = get_PDP_PFI_importance(decomp_grouped_sym)
bar([np.sqrt(I_PFI), np.sqrt(I_PDP)], grouped_features.print_names())
plt.title("Full background with grouping")
plt.show()

# %%
# Using regional backgrounds
# The groups are passed to the function
I_PDP, I_PFI = get_PDP_PFI_importance(decomp_grouped_sym, groups=groups)

fig, axes = plt.subplots(1, 2)
for i in range(2):

    bar([np.sqrt(I_PFI[i]), np.sqrt(I_PDP[i])], 
        grouped_features.print_names(),
        ax=axes[i],
        color=COLORS[i])
    axes[i].set_xlim(0, np.sqrt(I_PFI.max()))
    axes[i].set_title(rules[i], fontsize=10)
plt.show()

# %% [markdown]
# The feature importance of `pdays` and `poutcome` are
# different between the two regions. When `contact=unknown`,
# these two features are given no importance while they are
# top-3 and top-5 when `contact` $\in$ [`telephone`, `celullar`]
# We are about to see that this is because the two
# features do not vary when `contact=unknown`.
# %%
print(f"Pdays conditioned on contact=unknown={np.unique(background[~groups, 9])}")
print(f"Poutcome conditioned on contact=unknown={np.unique(background[~groups, 15])}")
# Poutcome=0 corresponds to 'unknown'

# %% [markdown]
# This is an example of insights on your model/data that can be provided by FDTrees.
# We are not 100% what `contact=unknown` represents in terms of the Bank data collected,
# but we can say with confidence that the samples with `contact=unknown` should be
# studied separately.
#
# Given the two regions, we can plot the regional ICE curves.
# We have to exclude the grouped feature `day:month` because ICE curves
# are one-dimensional. By setting `idxs=range(14)` in `partial_dependence_plot`, we tell
# the function to only plot the first 14 feature and exclude the 15th one, which is
# `day:month`.

# %%

partial_dependence_plot(decomp_grouped_sym, background, background, 
                        grouped_features, groups_method=tree.predict, rules=rules, idxs=range(14),
                        plot_hist=True, n_cols=4, figsize=(10, 10), normalize_y=True)
plot_legend(rules, ncol=2)
plt.show()

# %% [markdown]
# We often do not see the blue curve because the features do not vary when
# `contact=unknown`. To see the regional effects of `day:month`, we have to do separate scatter plots
# for both regions.
# %%
phi = decomp_grouped_sym[(14,)].mean(-1)
groups = groups.astype(bool)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
axes[0].set_title(rules[0])
axes[0].scatter(background[~groups, 5], background[~groups, 6], c=phi[~groups],
            norm=TwoSlopeNorm(vmin=phi.min(), vcenter=0, vmax=phi.max()),
            cmap='seismic')
axes[0].set_xlim(0.5, 31.5)
axes[0].set_ylim(-0.5, 11.5)
axes[0].set_xlabel("day")
axes[0].set_ylabel("month")

plt.figure()
axes[1].set_title(rules[1])
map = axes[1].scatter(background[groups, 5], background[groups, 6], c=phi[groups],
            norm=TwoSlopeNorm(vmin=phi.min(), vcenter=0, vmax=phi.max()),
            cmap='seismic')
axes[0].set_xlim(0.5, 31.5)
axes[0].set_ylim(-0.5, 11.5)
axes[1].set_xlabel("day")
axes[1].set_ylabel("month")
fig.colorbar(map, ax=axes.ravel().tolist())
plt.show()
