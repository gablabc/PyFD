# %% [markdown]
# # Linear Model on Adult Income
# In this tutorial, we will use PyFD to explain a linear model fitted to predict
# the income of US residents.
# %% 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from joblib import dump, load

from pyfd.data import get_data_adults
from pyfd.decompositions import get_components_linear
from pyfd.plots import setup_pyplot_font, bar, partial_dependence_plot
setup_pyplot_font(20)

# %% [markdown]
# We first load the data directly from PyFD and then fit a `LogisticRegression`
# model on top of a `Pipeline` that standardizes ordinal features and one-hot-encodes
# nominal ones
# %%

X, y, features = get_data_adults()
d = len(features)
features.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
# One Hot Encoding
ohe = ColumnTransformer([
                        ('id', StandardScaler(), features.ordinal),
                        ('ohe', OneHotEncoder(sparse_output=False), features.nominal)]
                        )
model = Pipeline([('ohe', ohe), ('predictor', LogisticRegression(max_iter=500))])
# Hyperparameter search space
C_grid = np.logspace(-3, 4, 20)
search = GridSearchCV(
    model,
    cv=KFold(),
    param_grid={"predictor__C": C_grid},
    verbose=2
)
search.fit(X_train, y_train)

# Recover the optimal CV model
model = search.best_estimator_
res = search.cv_results_
cv_perf = np.nan_to_num(res['mean_test_score'], nan=1e10)
best_idx = np.argmax(cv_perf)

# %%
plt.figure()
plt.scatter(1/C_grid, cv_perf, c='b', alpha=0.75)
plt.plot(1/C_grid[best_idx], cv_perf[best_idx], 'r*', markersize=10, markeredgecolor='k')
plt.xlabel("Regularisation 1/C")
plt.ylabel("Cross-Validated Accuracy")
plt.xscale('log')
plt.show()

# %%

# Pickle the model
dump(model, os.path.join("models", "adult_linear.joblib"))

# %% [markdown]
# Here we load the model. On subsequent reruns of the script,
# you do not need to retrain the model and can simply start by running
# this cell.
# %%
# Load the model
model = load(os.path.join("models", "adult_linear.joblib"))
print(model.score(X_test, y_test))

# %%[markdown]
## Local Feature Attributions
# Here we compute local feature attributions of the
# linear model. These attributions are then visualized as
# **Partial Dependence Plots**.
# %%

foreground = X_test
background = X_train
components = get_components_linear(model, foreground, background)

# %%
partial_dependence_plot(components, foreground, background, features, 
                        plot_hist=True, n_cols=6, figsize=(30, 15))
plt.show()

# %% [markdown]
# We next show the feature attributions of randomly chosen
# individuals from the dataset
# %% 

for idx in range(0, 20, 2):
    print(f"##### Instance {idx} #####")
    # Choose an instance
    x_explain = X[[idx]]
    x_map = features.print_value(x_explain.ravel())

    # Predictions and explanation
    pred = model.decision_function(x_explain)
    bar(np.array([components[(i,)][idx] for i in range(d)]), x_map)
    plt.title(f"Target {y[idx]} Logit {pred[0]:.3f}")
    plt.show()


# %%[markdown]
## Global Feature Importance
# Finally, we compute the global feature importance via
# $$ \Phi_i(h) = \sqrt{\mathbb{E}_{x\sim \mathcal{B}}[\phi_i(h, x)^2]}$$
# An important characteristic of PyFD is that global importance
# is computed by **aggregating** local attributions. This aggregation
# is left to the user.
# %%
FI = np.stack([(components[(i,)]**2).mean() for i in range(d)])
FI = np.sqrt(FI)
features_names = features.names()
bar(FI, features_names)
plt.show()
top_k = np.argsort(-FI)[:5]
print(f"Top 5 features {[features_names[i] for i in top_k]}")

# %% [markdown]
# When using the variance as an importance measure, we note
# `race` is not important. However, this is a bias of the measure
# because the data is very imbalanced. By doing another aggregation
# $$ \Phi_i(h) = \text{max}_{x}\,|\phi_i(h, x)|$$
# we get a different global picture.
# %%
FI = np.stack([np.abs(components[(i,)]).max() for i in range(d)])
FI = np.sqrt(FI)
features_names = features.names()
bar(FI, features_names)
plt.show()
top_k = np.argsort(-FI)[:5]
print(f"Top 5 features {[features_names[i] for i in top_k]}")

# %%
