"""
Test the LFA of a Spline Model on toy data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression

from pyfd.decompositions import get_components_linear, get_components_brute_force
from pyfd.features import Features

np.random.seed(42)
X = np.random.uniform(-1, 1, size=(2000, 2))
features = Features(X, ["x0", "x1"], ["num"]*2)
splt = SplineTransformer(n_knots=4, degree=3, include_bias=False)
H = splt.fit_transform(X)
y = H.dot(np.random.uniform(-2, 2, size=(H.shape[1], 1))) + 0.1*np.random.normal(0, 1, size=(2000, 1))

# Fit the model
model = LinearRegression()
model = Pipeline([('encoder', splt), ('predictor', model)])
model.fit(X, y.ravel())

line = np.linspace(-1, 1, 20)
foreground = np.column_stack((line, line))
decomposition_1 = get_components_linear(model, foreground, X, features)
decomposition_2 = get_components_brute_force(model.predict, foreground, X, features)

# Show PDP
for i in [0, 1]:
    plt.figure()
    plt.plot(line, decomposition_1[(i,)], 'k-o')
    plt.plot(line, decomposition_2[(i,)].mean(1)+0.05, 'r-o')
    plt.xlabel(f"$x{i}$")
    plt.ylabel(f"h{i} component")
plt.show()
