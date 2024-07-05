# %% [markdown]
# # Affine Invariance of linear model attributions.
# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
np.random.seed(42)

from pyfd.decompositions import get_components_linear

# %% [markdown]
# ## Linear Models
# We will start with the simplest model type : linear models 
# $h(x) = \omega_0 + \sum_{j=1}^d \omega_j x_j$,
# where $\omega_0$ is called the bias and $\omega_j$ are called the weights.
# Here is how you can fit and visualize a Linear Regression on a 1D toy dataset
# %%
X = np.random.uniform(1, 2, size=(100, 1))
x_explain = np.array([1.1])
y = 2 - X.ravel() + 0.1 * np.random.normal(size=(100,))
model = LinearRegression().fit(X, y)

plt.figure()
plt.scatter(X.ravel(), y)
plt.vlines(0, 0, 2.2, 'k')
line = np.linspace(0, 2, 10)
plt.plot(line, model.intercept_ + model.coef_[0]*line, 'k--', linewidth=5)
plt.scatter(x_explain, model.intercept_ + model.coef_[0]*x_explain, marker="*", c='r', s=200, zorder=3)
plt.xlim(-0.1, 2)
plt.ylim(0, 2.2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% [markdown]
# Now lets say you want to understand the prediction at a point $x_\text{explain}=1.1$ (indicated by a red star). 
# You might be tempted to interpret the value $\omega \,x_\text{explain}$ as the importance of $x_\text{explain}$ 
# for the prediction. This yields
# %%
print(f"Local Attribution of x_explain : {float(model.coef_[0]*x_explain[0]):.2f}")
# %% [markdown]
# But if we center the dataset, then we get an explanation with a completely different sign!
# %%
X_mean = X.mean(0)
X_std = X.std(0)
# Centered data X_
X_ = (X - X_mean) / X_std
x_explain_ = (x_explain - X_mean) / X_std
model_ = LinearRegression().fit(X_, y)

plt.figure()
plt.scatter(X_.ravel(), y)
plt.vlines(0, -2, 2, 'k')
line = np.linspace(-2, 2, 10)
plt.plot(line, model_.intercept_ + model_.coef_[0]*line, 'k--', linewidth=5)
plt.scatter(x_explain_, model_.intercept_ + model_.coef_[0]*x_explain_, marker="*", c='r', s=200, zorder=3)
plt.xlim(-2, 2)
plt.ylim(-0.2, 1.2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print(f"Local Attribution of x_explain_ : {float(model_.coef_[0]*x_explain_[0]):.2f}")

# %% [markdown]
# What is happening here? The issue is that the importance $\omega x_\text{explain}$ is not
# **invariant** to affine mappings of the feature. Lets say $x$ represents a temperature. 
# This means that changing the units from Celcius to Fareinheit, will change the
# Local Feature Attribution for the same instance in the data. To solve this problem, we must
# not think of importance in absolute terms but in **relative** terms. Insteads of presenting the
# impact of $x$ on a specific prediction $h(x)$, we must investigate the impact of $x$ on the
# difference between $h(x)$ and a baseline prediction $\mathbb{E}_{z\sim\mathcal{B}}[h(z)]$
# where $\mathcal{B}$ is called the background distribution. Simply put, the reference to which we
# compare our prediction is the average prediction over some distribution chosen by the user.
# This background distribution $\mathcal{B}$ is an essential element of PyFD and so it
# is important to get an early intuition on its role in providing explanations.
# When considering explanations relative to a baseline, the Local Feature Importance becomes 
# $\omega\, (x-\mathbb{E}_{z\sim\mathcal{B}}[z])$ which is invariant to any affine mapping on $x$.
# %%
# Original data
foreground = x_explain.reshape((1, -1))
background = X
decomposition = get_components_linear(model, foreground, background)
print(f"Local Attribution of x_explain : {decomposition[(0,)][0]:.2f}")

# Standardized data
foreground = x_explain_.reshape((1, -1))
background = X_
decomposition = get_components_linear(model_, foreground, background)
print(f"Local Attribution of x_explain_ : {decomposition[(0,)][0]:.2f}")

# %% [markdown]
# ## Spline Models
# Let us complexity the problem so that a linear model is no longer adequate.
# %%
y = np.log(X-0.99).ravel() + 0.1 * np.random.normal(size=(100,))
plt.figure()
plt.scatter(X.ravel(), y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% [markdown]
# To handle this task, we will need to have a more expressive model :
# $h(x) = \omega_0 + \sum_{j=1}^d h_j(x_j)$,
# which is called an additive model. To represent the
# individual functions $h_j$, we define a spline basis
# along each feature
# %%
splines = SplineTransformer(n_knots=3).fit(X)
line = np.linspace(X.min(), X.max(), 100).reshape((-1, 1))
H = splines.transform(line)
# Plot the basis functions
plt.figure()
plt.plot(line, H, linewidth=4)
plt.xlabel("x")
plt.ylabel("Spline Basis")
knots = splines.bsplines_[0].t
plt.vlines(knots[3:-3], ymin=0, ymax=1.05, linestyles="dashed", color="k")
plt.ylim(0, 1)
plt.show()

# %% [markdown]
# These basis could be used as a new feature matrix that can be send to a linear model
# e.g. `linear_model.fit(H, y)`. However, this introduces an issue of overparametrization. 
# Indeed, the splines sum up to 1 at each $x$ and so the bias term $\omega_0$ is redundant. We have two options:
# - Ignore the bias term in the regression. This will not be sufficient when there are various input features
# - Remove one of the splines.
# %%
splines = SplineTransformer(include_bias=False, n_knots=3).fit(X)
H = splines.transform(line)
# Plot the basis functions
plt.figure()
plt.plot(line, H, linewidth=4)
plt.xlabel("x")
plt.ylabel("Reduced Spline Basis")
knots = splines.bsplines_[0].t
plt.vlines(knots[3:-3], ymin=0, ymax=1.05, linestyles="dashed", color="k")
plt.ylim(0, 1)
plt.show()

# %% [markdown]
# Note that the last basis has been removed to account for overparametrization.
# The key with PyFD is that the attribution is invariant to the
# choice of either option 1 or 2.
# %%
foreground = x_explain.reshape((1, -1))
background = X

# Option 1
model_1 = Pipeline([('spline', SplineTransformer(include_bias=True)), 
                    ('predictor', LinearRegression(fit_intercept=False)) ]).fit(X, y)
decomp_1 = get_components_linear(model_1, foreground, background)
print(f"Local Attribution of x_explain : {decomp_1[(0,)][0]:.2f}")

# %%
# Option 2
model_2 = Pipeline([('spline', SplineTransformer(include_bias=False)), 
                    ('predictor', LinearRegression(fit_intercept=True)) ]).fit(X, y)
decomp_2 = get_components_linear(model_2, foreground, background)
print(f"Local Attribution of x_explain : {decomp_2[(0,)][0]:.2f}")

# %%

plt.figure()
line = np.linspace(1, 2, 10)
plt.scatter(X.ravel(), y)
plt.plot(line, model_1.predict(line.reshape((-1, 1))), "k--", linewidth=5)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %%
