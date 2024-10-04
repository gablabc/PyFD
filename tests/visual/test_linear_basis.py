"""
Test fitting a linear model on basis functions
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer
from sklearn.preprocessing import KBinsDiscretizer

from pyfd.plots import setup_pyplot_font
from pyfd.decompositions import get_components_linear
from pyfd.features import Features

np.random.seed(42)
setup_pyplot_font(20)

# Generate Data
def generate_data(n_samples):
    X = np.random.uniform(0.1, 2, size=(n_samples, 1))
    y = np.log(X) + 0.25 * np.random.normal(size=(n_samples,1 ))
    features = Features(X, ["x0"], ["num"])
    return X, y.ravel(), features

X, y, features = generate_data(500)
line = np.linspace(0.1, 2, 100)

# Basic linear
model = Ridge().fit(X, y)
decomposition_1 = get_components_linear(model, line, X, features)
plt.figure()
plt.plot(line, decomposition_1[(0,)], 'k-', linewidth=5)
plt.scatter(X, y-y.mean(), alpha=0.5)
plt.xlabel("input")
plt.ylabel("Additive component")
plt.title("Linear Basis")


# Log-basis
model = Pipeline([('log', FunctionTransformer(np.log)), ('predictor', Ridge())]).fit(X, y)
decomposition_1 = get_components_linear(model, line, X, features)
plt.figure()
plt.plot(line, decomposition_1[(0,)], 'k-', linewidth=5)
plt.scatter(X, y-y.mean(), alpha=0.5)
plt.xlabel("input")
plt.ylabel("Additive component")
plt.title("Log Basis")


# Step-basis
model = Pipeline([('bins', KBinsDiscretizer(encode='onehot-dense')), ('predictor', Ridge())]).fit(X, y)
decomposition_1 = get_components_linear(model, line, X, features)
plt.figure()
plt.plot(line, decomposition_1[(0,)], 'k-', linewidth=5)
plt.scatter(X, y-y.mean(), alpha=0.5)
plt.xlabel("input")
plt.ylabel("Additive component")
plt.title("Step Basis")


# Spline-basis
model = Pipeline([('spline', SplineTransformer(include_bias=False)), ('predictor', Ridge())]).fit(X, y)
decomposition_1 = get_components_linear(model, line, X, features)
plt.figure()
plt.plot(line, decomposition_1[(0,)], 'k-', linewidth=5)
plt.scatter(X, y-y.mean(), alpha=0.5)
plt.xlabel("input")
plt.ylabel("Additive component")
plt.title("Spline Basis")


plt.show()
