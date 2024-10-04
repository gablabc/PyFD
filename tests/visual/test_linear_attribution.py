"""
Test the LFA of a Linear Model on toy data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression

from pyfd.plots import setup_pyplot_font
from pyfd.decompositions import get_components_brute_force, get_components_linear
from pyfd.features import Features

np.random.seed(42)
setup_pyplot_font(20)

# Generate Data
def generate_data(n_samples, noise, task="regression"):
    Z = np.random.normal(size=(n_samples, 2))
    X = Z.dot(np.array([[1, 2], [-0.5, 0.25]]))
    y = 2 * X[:, [0]] + X[:, [1]] + noise * np.random.normal(size=(n_samples,1))
    if task == "classification":
        y = (y > 0).astype(int)
    features = Features(X, ["x0", "x1"], ["num", "num"])
    return X, y.ravel(), features

# Main
noise = 1
model_type = {"regression" : Ridge(), "classification" : LogisticRegression()}
for task in ["regression", "classification"]:
    print(f"Doing {task}")
    X, y, features = generate_data(5000, noise, task=task)
    model = model_type[task].fit(X, y)

    line = np.linspace(-3, 3, 10)
    foreground = np.column_stack((line, line))
    decomposition_1 = get_components_linear(model, foreground, X, features)
    if task == "classification":
        decomposition_2 = get_components_brute_force(model.decision_function, foreground, X, features)
    else:
        decomposition_2 = get_components_brute_force(model.predict, foreground, X, features)

    # Show PDP
    for i in [0, 1]:
        plt.figure()
        plt.plot(line, decomposition_1[(i,)], 'k-o')
        plt.plot(line, decomposition_2[(i,)].mean(1)+0.2, 'r-o')
        plt.xlabel(f"$x{i}$")
        plt.ylabel(f"h{i} component")
        plt.ylim(-10, 10)
    plt.show()
