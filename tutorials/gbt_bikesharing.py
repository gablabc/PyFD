# %% [markdown]
# # Introduction to the PyFD package
# This tutorial introduces most of the functionalities of the `PyFD` package.
# We will go quickly through the code and leave in-depth explanations for subsequent tutorials.
# In this example, we will decompose a complex GBT model trained to predict bike rentals. We will see how to 
# interpret said model in spite of the presence of strong feature interactions.
# %% 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Local imports
from pyfd.data import get_data_bike
from pyfd.plots import setup_pyplot_font

setup_pyplot_font(15)

# %% [markdown]
# ## Fitting model
# First load the data and a `Features` object that stores
# information about the various features. Note that the data **X** must **always**
# be a numerical numpy array. Categorical features are assumed to have been ordinally encoded.
# %%

X, y, features = get_data_bike()
features.feature_objs[1].cats = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
features.feature_objs[4].cats = ["Mon", "Thu", "Wed", "Thu", "Fri", "Sat", "Sun"]
features.summary()

# %% [markdown]
# ## Fitting model
# Having fetched the data, we fit a `HistGradientBoostingRegressor`.
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = HistGradientBoostingRegressor(random_state=0)

# %%
grid = {"learning_rate": np.logspace(-2, 0, 10),
        "max_depth" : [3, 4, 5, 6, 7],
        "max_iter" : [50, 100, 150, 200],
        'min_samples_leaf' : [1, 20, 40, 60, 80, 100]}

search = RandomizedSearchCV(
    model,
    cv=KFold(),
    scoring='neg_root_mean_squared_error',
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
cv_perf = np.nan_to_num(-res['mean_test_score'], nan=1e10)
best_idx = np.argmin(cv_perf)

# %%

plt.figure()
plt.scatter(res['param_learning_rate'], cv_perf, c='b', alpha=0.75)
plt.plot(res['param_learning_rate'][best_idx], cv_perf[best_idx], 'r*', markersize=10, markeredgecolor='k')
plt.xlabel("Learning Rate")
plt.ylabel("Cross-Validated RMSE")
plt.xscale('log')
# plt.ylim(97, 120)
plt.show()

# %%

# Pickle the model
dump(model, os.path.join("models", "bike_gbt.joblib"))

# %%
# Load the model
model = load(os.path.join("models", "bike_gbt.joblib"))
test_preds = model.predict(X_test)
print(mean_squared_error(test_preds, y_test, squared=False))

# %%[markdown]
# ## Additive Decomposition
# This models appears to make accurate predictions of bike rentals.
# But do you trust it? The issue it that the model $h$ is a black-box and
# we do not (yet) understand the mechanisms happening inside. To get
# more insight on our model, we are going to compute a functional
# decomposition. That is, given a reference data distribution $\mathcal{B}$
# we aim at expressing $h$ as the following sum
# $$h(x) = \sum_{u\subseteq [d]}h_{u,\mathcal{B}}(x)$$
# where $h_{u,\mathcal{B}}$ only depends on the subvector $x_u$.
# Let's visualize the additive terms $h_{i,\mathcal{B}}$ of the decomposition
# which only depend on feature $x_i$.
# %%
from pyfd.decompositions import get_components_tree, get_CoE
from pyfd.plots import partial_dependence_plot

# Reference data
background = X_train
# We evaluate the decomp on the whole test set
foreground = X_test
# Compute the functional decomposition
decomp = get_components_tree(model, foreground, background, algorithm="leaf")
# Plot the additive terms
partial_dependence_plot(decomp, foreground, background, features,
                        plot_hist=True, n_cols=5, figsize=(24, 10))
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
# strong feature interactions. However, for this model, the Cost of Exclusion (CoE)
# $$\mathbb{E}[(h(x) - h_{\text{add}}(x))^2] / \mathbb{V}[h(x)]$$ 
# is large.
# %%
print(f"CoE : {get_CoE(decomp, anchored=False, foreground_preds=test_preds):.2f} %")

# %% [markdown]
# 30% of the variability in the model cannot be
# explained by its additive approximation and so the curves
# shown previously are not representative of the true model
# behavior. We will now present various ways of dealing with
# this issue.

# %% [markdown]
# ## Dealing with Interactions
# ### Shapley-Taylor Indices
# A first solution to deal with feature interactions is to compute the Shapley-Taylor indices
# $$\Phi_{ij}(h, x) := 
#   \begin{cases}
#         h_i(x) & \text{if  } i=j\\
#         \sum_{u:\{i,j\}\subseteq u} h_{u}(x) / {|u| \choose 2} & \text{otherwise},
#   \end{cases}$$
# which are pair-wise interaction indices that take $|u|$-way interactions and share them evenly between 
# all pairs of features involved. These indices can be computed efficiently for tree-based models.
# %%
from pyfd.shapley import taylor_treeshap
from pyfd.plots import interactions_heatmap, plot_interaction

# This implementation uses the `recurse` algorithm which needs
# a small background
background = X_train[:500]
Phis = taylor_treeshap(model, background, background)
print(Phis.shape)

# %% [markdown]
# Given these indices, we can compute a heatmap of the
# pairwise interaction strenght among the features
# %%

interactions_heatmap(Phis, features.names())
plt.show()

# %% [markdown]
# We observe strong interactions between hour-workingday,
# hour-temp, and hour-year. These interactions can be visualized
# as functions of the features involved.
# %% HOUR versus YEAR

plot_interaction(2, 0, background, Phis, features)
plt.show()

# %% [markdown]
# According to this plot, there seems to be ~30 more bike 
# rentals in 2012 at rush hours compared to 2011.
# This effect cannot be captured by an additive model.
# %% HOUR versus WORKINGDAY 

plot_interaction(2, 5, background, Phis, features)
plt.show()

# %% [markdown]
# This interaction is interesting, we see that in non-working
# days, there are more bike rentals in the afternoon between 10h and 
# 16h. This may be because people are more likely to bike
# for pleasure during this period. During workingdays,
# the bikes are most likely rented to go to work.
# %% HOUR versus TEMPERATURE

plot_interaction(2, 7, background, Phis, features)
plt.show()

# %% [markdown]
# There are two interesting patterns in this figure.
# - First, early in the morning, the effect of increasing
# temperature is reduced, which would mean that early bikers are
# less influenced by temperature. 
# - Second, between 15-20h, the positive effect of rising temperature on
# bike rentals is amplified. This could be because hot
# temperature encourages both work-related and recreational biking.

# %% [markdown]
# ### Grouping Interacting Features
# We saw how to compute and interpret Shapley-Taylor indices.
# One of their weaknesses is that they only quantify pair-wise
# interactions and so they cannot help understanding a possible
# three-way interaction between hour-temp-workingday. To
# understand the joint effect of these variables on the model,
# on could group them into a single **meta-feature**.
# %%
from pyfd.fd_trees import CoE_Tree
from pyfd.decompositions import get_components_tree, get_CoE
from pyfd.shapley import interventional_treeshap
from pyfd.plots import bar

grouped_features = features.group([[2, 5, 7]])
grouped_features.summary()

# %% [markdown]
# We have a new feature `hr:workingday:temp`. The column `I^-1({i})` in the 
# summary table represents the pre-image of each feature i.e. what columns of *X* 
# map to each feature. Here we have `I^-1(7)=[2, 5, 7]` because columns 2, 5 and 7 
# all encode the 7th feature `hr:workingday:temp`.
#
# To compute a decomposition using this new meta-feature, the `Imap_inv` attribute 
# of `grouped_features` must be passed to the the decomposition algorithm.
# %%

background = X_train
foreground = X_test
# Without feature grouping
decomp = get_components_tree(model, foreground, background, algorithm="leaf")
# With feature grouping
decomp_grouped = get_components_tree(model, foreground, background, algorithm="leaf", 
                                    Imap_inv=grouped_features.Imap_inv)
print(f"CoE : {get_CoE(decomp_grouped, anchored=False, foreground_preds=test_preds):.2f} %")

# %% [markdown]
# We have reduced the CoE by a factor three by grouping these features 
# together. The downside is that understanding
# the effect of the meta-feature `hr:workingday:temp` requires
# visualizing the coresponding model component $h_{i,\mathcal{B}}(x)$ as 3D
# plot. Fortunatelly, since `workingday` is binary
# we only need to make two 2D scatter plots.
# %%
phi = decomp_grouped[(7,)]

plt.figure()
plt.title("working day = True")
idx = foreground[:, 5]==1
plt.scatter(foreground[idx, 2], foreground[idx, 7], c=phi[idx])
plt.xlabel("hr")
plt.ylabel("temp")

plt.figure()
plt.title("working day = False")
plt.scatter(foreground[~idx, 2], foreground[~idx, 7], c=phi[~idx])
plt.xlabel("hr")
plt.ylabel("temp")

plt.show()

# %% [markdown]
# Shapley values can also be computed while treating these tree features as a single
# **player** in a coallitional game. This is done by passing the 
# `Imap_inv` attribute to `interventional_treeshap`.
# %%
# Without feature grouping
shapley_values = interventional_treeshap(model, foreground, background, algorithm="leaf")
# With feature grouping
shapley_values_grouped = interventional_treeshap(model, foreground, background, 
                                            Imap_inv=grouped_features.Imap_inv, algorithm="leaf")

# %% Local Feature Attributions

d = len(features)
D = len(grouped_features)
for idx in range(0, 10, 2):
    print(f"##### Instance {idx} #####")
    # Choose an instance
    x_explain = foreground[[idx]]

    # Without feature grouping
    pred = model.predict(x_explain)
    x_map = features.print_value(x_explain.ravel())
    pdp = np.array([decomp[(i,)][idx] for i in range(d)])
    bar([pdp, shapley_values[idx]], x_map)
    plt.title(f"Target {y_test[idx]} Prediction {pred[0]:.3f}")
    # The disagreement between PDP-SHAP
    print(np.sqrt(np.mean((pdp - shapley_values[idx])**2)))

    # With feature grouping
    x_map = grouped_features.print_value(x_explain.ravel())
    pdp_grouped = np.array([decomp_grouped[(i,)][idx] for i in range(D)])
    bar([pdp_grouped, shapley_values_grouped[idx]], x_map)
    plt.title(f"Target {y_test[idx]} Prediction {pred[0]:.3f}")
    # The disagreement between PDP-SHAP
    print(np.sqrt(np.mean((pdp_grouped - shapley_values_grouped[idx])**2)))
    plt.show()

# %% Global Feature Importance

# Without feature grouping
I_PDP = np.sqrt( np.stack([(decomp[(i,)]**2).mean() for i in range(d)]) )
I_SHAP = np.sqrt(np.mean(shapley_values**2, axis=0))
features_names = features.names()
bar([I_SHAP, I_PDP], features_names)

# With feature grouping
I_PDP = np.sqrt( np.stack([(decomp_grouped[(i,)]**2).mean() for i in range(D)]) )
I_SHAP = np.sqrt(np.mean(shapley_values_grouped**2, axis=0))
features_names = grouped_features.names()
bar([I_SHAP, I_PDP], features_names)
plt.show()

# %% [markdown]
# We note that there are still interactions involving the
# group `hr:workingday:temp` and other features. Hence we could add 
# another feature (e.g. `yr`) to the group. Nonetheless, grouping
# features lead to a loss of granularity of the decomposition.
# For example, we could put all features into one group and the
# explanations would become useless. We present alternatives
# that make explanations more granular.
# %% [markdown]
# ### GADGET-PDP
# This method consists of computing multiple $h_{i,\mathcal{B}}$ 
# curves per feature using data $\mathcal{B}$ that lands on specific regions of the input space.
# These regions are described by rules involving the remaining features. 
# This technique requires computing anchored decompositions. 
# %%
from pyfd.decompositions import get_components_tree
from pyfd.plots import partial_dependence_plot

# Compute an anchored decomposition requires smaller backgrounds
# We also need foreground=background to train FDTrees
background = X_train[:500]
decomp = get_components_tree(model, background, background, anchored=True)

# %% [markdown]
# GADGET-PDP is automatically called when passing the argument
# `grouping="gadget-pdp` to `partial_dependence_plot`.
# %%

partial_dependence_plot(decomp, background, background, features,
                        groups_method="gadget-pdp",
                        fd_trees_kwargs={"max_depth":2, "alpha": 0.01}, 
                        n_cols=3, figsize=(20, 20))
plt.show()

# %% [markdown]
# For every feature, we plot multiple curves based on regions
# computed with the remaining features. For instances, the curves
# of `hr` are partitioned according to the values of the features
# `temp` and `workingday`. Given these plots, one can understand how 
# the effect of perturbing a **single** feature depends on the fixed value of the 
# others.
#
# The $\alpha$ parameter that is passed to the GADGET-PDP regularizes the complexity
# the trees. By reducing it, we allow for more regions and by increasing
# it we reduce the number of regions.
# %%
# Less regions
partial_dependence_plot(decomp, background, background, features, 
                        groups_method="gadget-pdp", 
                        fd_trees_kwargs={"max_depth":2, "alpha": 0.02},
                        normalize_y=False, n_cols=3, figsize=(20, 20))
plt.show()

# %%
# More regions
partial_dependence_plot(decomp, background, background, features,
                        groups_method="gadget-pdp", 
                        fd_trees_kwargs={"max_depth":2, "alpha": 0.001},
                        normalize_y=False, n_cols=3, figsize=(20, 20))
plt.show()

# %% [markdown]
# ### Regional Explanations with FDTrees
# Instead of fitting a different tree for each feature, we can fit a single tree and
# explain the model behavior on each separate region.
# %%
from pyfd.fd_trees import CoE_Tree
from pyfd.decompositions import get_PDP_PFI_importance, get_components_tree
from pyfd.shapley import interventional_treeshap
from pyfd.plots import bar, attrib_scatter_plot, plot_legend, COLORS

# Compute an anchored decomposition requires smaller backgrounds
# FD-Trees require background=foreground
background = X_train[:1000]
decomp = get_components_tree(model, background, background, anchored=True)

# %%
# Fit a FDTree by passing the `decomposition` as argument.
interacting_features = [0, 2, 5]
tree = CoE_Tree(max_depth=2, features=features.select(interacting_features), alpha=0.02)
tree.fit(background[:, interacting_features], decomp)
tree.print(verbose=True)

# %% [markdown]
# This FD-Tree has five leaves based on time and workingday.
# Crucially, like we managed to do previously by grouping features, 
# we are able to reduce the CoE to ~9%. Hence, the model can be
# better approximated as additive if we stay on the separate leaves.
#
# Given the FD-Tree, we visualize the global importance over its leaves.
# %%

# Using the whole background
I_PDP, I_PFI = get_PDP_PFI_importance(decomp)
bar([np.sqrt(I_PFI), np.sqrt(I_PDP)], features.names())
plt.show()

# %% [markdown]
# Note the strong disagreements regarding the importance of
# `workingday` : PFI ranks it 2 while PDP ranks it last.
# Because this feature interacts with others, its feature importance
# is highly uncertain.
# %%
# Using regional backgrounds
groups = tree.predict(background[:, interacting_features])
rules = tree.rules(use_latex=True)
# The groups are passed to the function
I_PDP, I_PFI = get_PDP_PFI_importance(decomp, groups=groups)

fig, axes = plt.subplots(2, 2)
for i in range(4):
    row = i // 2
    col = i % 2

    bar([np.sqrt(I_PFI[i]), np.sqrt(I_PDP[i])], 
        features.names(),
        ax=axes[row][col],
        color=COLORS[i])
    axes[row][col].set_xlim(0, np.sqrt(I_PFI.max()))
    axes[row][col].set_title(rules[i], fontsize=10)
plt.show()

# %% [markdown]
# Let us take the time to analyze these results. First,
# there are no longer disagreements regarding the importance
# of `workingday` : all regions give it no importance. This
# does not mean that `workingday` is not important!
# It simply means that attributing an score to `workingday` is
# unreliable and it is best to interpret it as a
# feature that determines optimal regions.
#
# Secondly, the `hr` feature is always the most important
# regardless of the region. This add weight to the argument that
# this is **the most** important feature.
#
# We can also compute regional shapley values and plot
# them along side the regional $h_{i,\mathcal{B}}$ curves.
# %%
# Compute SHAP values on each separate leaves
phis_list = [0] * tree.n_groups
for group_idx in range(tree.n_groups):
    print(f"### Region {group_idx} ###")
    idx_select = (groups == group_idx)
    subset_background = background[idx_select]

    # SHAP
    phis_list[group_idx] = interventional_treeshap(model,
                                        subset_background, 
                                        subset_background, 
                                        algorithm='leaf')

# %%

attrib_scatter_plot(decomp, phis_list, background, features, 
                    groups=groups, normalize_y=False, figsize=(24, 10))
plot_legend(rules, ncol=2)
plt.show()

# %%
