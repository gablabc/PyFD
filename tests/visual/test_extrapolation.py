""" Toy Example with Interactions """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from pyfd.extrapolation import sample_synthetic_points

# Generate the data
np.random.seed(42)
rho = 0.99
cov = np.array([[1, rho], [rho, 1]])
X = np.random.multivariate_normal([0, 0], cov, size=(1000,))
outlier = IsolationForest().fit(X)
Z = sample_synthetic_points(X, X)
print(f"Extrapolation Score with no regions : {outlier.decision_function(Z).mean():.2f}\n")

# Predictions on a grid
XX, YY = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
ZZ = outlier.decision_function(np.column_stack((XX.ravel(), YY.ravel()))).reshape(XX.shape)


plt.figure()
varmap = plt.contourf(XX, YY, ZZ, cmap='Reds', alpha=0.25)
plt.scatter(X[:, 0], X[:, 1], c='b', marker="o")
plt.scatter(Z[:, 0], Z[:, 1], c='r', marker="+")
plt.colorbar(varmap)



# 4 regions
groups_method = lambda X : np.digitize(X[:, 0], bins=[-1.5, 0, 1.5])
Z = sample_synthetic_points(X, X, groups_method=groups_method)
print(f"Extrapolation Score with 4 regions : {outlier.decision_function(Z).mean():.2f}\n")

ZZ = outlier.decision_function(np.column_stack((XX.ravel(), YY.ravel()))).reshape(XX.shape)

plt.figure()
varmap = plt.contourf(XX, YY, ZZ, cmap='Reds', alpha=0.25)
plt.scatter(X[:, 0], X[:, 1], c='b', marker="o")
plt.scatter(Z[:, 0], Z[:, 1], c='r', marker="+")
plt.colorbar(varmap)
plt.vlines([-1.5, 0, 1.5], X[:, 1].min(), X[:, 1].max(), colors='k', linewidth=2)



# 6 regions
groups_method = lambda X : np.digitize(X[:, 0], bins=[-2, -1, 0, 1, 2])
Z = sample_synthetic_points(X, X, groups_method=groups_method)
print(f"Extrapolation Score with 6 regions : {outlier.decision_function(Z).mean():.2f}\n")

ZZ = outlier.decision_function(np.column_stack((XX.ravel(), YY.ravel()))).reshape(XX.shape)

plt.figure()
varmap = plt.contourf(XX, YY, ZZ, cmap='Reds', alpha=0.25)
plt.scatter(X[:, 0], X[:, 1], c='b', marker="o")
plt.scatter(Z[:, 0], Z[:, 1], c='r', marker="+")
plt.colorbar(varmap)
plt.vlines([-2, -1, 0, 1, 2], X[:, 1].min(), X[:, 1].max(), colors='k', linewidth=2)



# Feature grouping
Z = sample_synthetic_points(X, X, Imap_inv=[[0, 1]])
print(f"Extrapolation Score with grouping : {outlier.decision_function(Z).mean():.2f}\n")

ZZ = outlier.decision_function(np.column_stack((XX.ravel(), YY.ravel()))).reshape(XX.shape)

plt.figure()
varmap = plt.contourf(XX, YY, ZZ, cmap='Reds', alpha=0.25)
plt.scatter(X[:, 0], X[:, 1], c='b', marker="o")
plt.scatter(Z[:, 0], Z[:, 1], c='r', marker="+")
plt.colorbar(varmap)
plt.vlines([-2, -1, 0, 1, 2], X[:, 1].min(), X[:, 1].max(), colors='k', linewidth=2)
plt.show()
