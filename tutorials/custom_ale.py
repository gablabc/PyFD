# %% [markdown]
# # Custom Implementation of ALE
# This tutorial presents how to compute Accumulated Local Effects (ALE) using PyFD. 
# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from pyfd.decompositions import get_components_brute_force
from pyfd.utils import get_quantiles
from pyfd.plots import setup_pyplot_font, partial_dependence_plot
from pyfd.features import Features
from pyfd.extrapolation import sample_synthetic_points

setup_pyplot_font(20)

# %% [markdown]
# We first generate correlated features and a ground truth that only depends on $x_2$
# %%

n_samples = 300
X = np.random.multivariate_normal([0, 0],
                                  np.array([[1, 0.99], [0.99, 1]]),
                                  size=(n_samples,))
X = norm.cdf(X)
features = Features(X, [r"$x_1$", r"$x_2$"], ["num", "num"])
h_star = lambda X: X[:, 1]
y = h_star(X) + np.sqrt(0.01) * np.random.normal(size=(n_samples,))

XX, YY = np.meshgrid(np.linspace(0, 1, 100), 
                     np.linspace(0, 1, 100))
ZZ = h_star( np.column_stack((XX.ravel(), YY.ravel())) ).reshape(XX.shape)

plt.figure()
plt.contourf(XX, YY, ZZ, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c='b', s=15)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()

# %% [markdown]
# Now, in real life you do not have access to the model that generated the labels and hence you
# would estimate it with a machine learning model.
# %%
model = RandomForestRegressor(max_depth=3, min_samples_leaf=20)
model.fit(X[:250], y[:250])
print(f"R2 score on test : {100*model.score(X[:250], y[:250]):.2f}%")
print(f"R2 score on test : {100*model.score(X[250:], y[250:]):.2f}%")
ZZ = model.predict( np.column_stack((XX.ravel(), YY.ravel())) ).reshape(XX.shape)

plt.figure()
plt.contourf(XX, YY, ZZ, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c='b', s=15)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()

# %% [markdown]
# Since this model fits the labels pretty well, you might be tempted to explain it using `pyfd` to gain
# some insight on the data-generating mechanism. Computing ICE curves yields
# %%
# Compute the additive decomposition of the model
decomposition = get_components_brute_force(model.predict, X, X)
# Plot results for each feature
partial_dependence_plot(decomposition, X, X, features)
plt.show()

# %% [markdown]
# We note a strong dependence on $x_1$ despite the fact that the 
# ground truth does not utilize said feature. Why is this happening?
# The reason is that anchored decompositions extrapolate the data.
# This can be verified by calling the `sample_synthetic_points`
# function that generates fake data samples used to compute the
# anchored/interventional decompositions.
# %%
fake_X = sample_synthetic_points(X, X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='b')
plt.scatter(fake_X[:, 0], fake_X[:, 1], c='r', marker='+')
plt.show()

# %% [markdown]
# As we extrapolate the data, we evaluate the model on regions where its
# output is not trusthworthy nor representative of the ground-truth
# function. To solve this issue, one can implement the ALE method which
# works as follows
# - Bin feature $j$ with quantiles, define $G_k$ as the set of all
# datapoints that land in bin $k$.
# - Compute the interventional component $h_{j, G_k}$ on each bin
# $k$.
# - Visualize the results.
# %%
from pyfd.plots import COLORS
for j in range(2):
    # Compute quantiles along feature j
    quantiles, n_bins = get_quantiles(X[:, j], 6)
    # Compute the groups G_k for bin k
    def binner(X):
        return np.clip(np.digitize(X[:, j], quantiles, right=True) - 1, 0, None)
    bin_index = binner(X)
    plt.figure()
    # For each bin we compute a separate functional decomposition
    for k in range(n_bins):
        G = X[bin_index==k]
        # We evaluate the decomposition on a simple line spanning the bin
        foreground = np.linspace(quantiles[k], quantiles[k+1], 5)
        # Compute the decomposition using the PyFD API
        decomp = get_components_brute_force(model.predict, foreground, G, Imap_inv=[[j]])
        ice_curves = decomp[(0,)] + decomp[()]
        pdp_curves = ice_curves.mean(1)
        ice_curves += - ice_curves.mean(0) + ice_curves.mean()
        plt.plot(foreground, ice_curves, COLORS[k%5], alpha=0.05)
        plt.plot(foreground, pdp_curves, COLORS[k%5])
    
    y_min, y_max = plt.gca().get_ylim()
    plt.vlines(quantiles, y_min, y_max, colors='k')
    plt.ylim(y_min , y_max)
    plt.ylabel("ALE")
    plt.xlabel(r"$x_" + str(j+1) +"$")
plt.show()

# %% [markdown]
# Looking at these figures, we note that near the center
# of the data, the dependence on $x_1$ is very weak compared
# to $x_2$. This is not necessarily true near the edges of the
# data. We also observe that ALE leads to less extrapolation than
# the ICE/PDP curves computed previously.
# %%

for j in range(2):
    # Compute quantiles along feature j
    quantiles, n_bins = get_quantiles(X[:, j], 6)
    # Compute the groups G_k for bin k
    def binner(X):
        return np.clip(np.digitize(X[:, j], quantiles, right=True) - 1, 0, None)
    groups_method = lambda X : np.clip(np.digitize(X[:, j], quantiles, right=True) - 1, 0, None)
    fake_X = sample_synthetic_points(X, X, groups_method, Imap_inv=[[j]])
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c='b')
    plt.scatter(fake_X[:, 0], fake_X[:, 1], c='r', marker='+')
    if j == 0:
        y_min, y_max = plt.gca().get_ylim()
        plt.vlines(quantiles, y_min, y_max, colors='k')
    else:
        x_min, x_max = plt.gca().get_xlim()
        plt.hlines(quantiles, x_min, x_max, colors='k')
plt.show()

# %% [markdown]
# We end this tutotial by noting that there is no
# dedicated `ALE` function in PyFD. Rather, user have to
# build ALE themselves by binning the feature, grouping the
# data instances, and computing the functional decomposition in
# each region. We believe leaving this work to the user increases
# the flexibility and transparency of ALE computations.
# %%