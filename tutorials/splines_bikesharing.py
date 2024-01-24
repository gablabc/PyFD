# %% [markdown]
# # Spline Additive Model on BikeSharing
# In this tutorial, we will use PyFD to explain an additive model fitted to predict
# the number of bike rentals given time and weather features.
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from joblib import dump, load

from pyfd.data import get_data_bike
from pyfd.decompositions import get_components_linear
from pyfd.plots import setup_pyplot_font, bar, partial_dependence_plot
setup_pyplot_font(20)

# %% [markdown]
# We first load the data directly from PyFD and then fit a `Ridge` model
# model on top of a `Pipeline` that does the following
# - Keep the features `yr`, `holiday`, `workingday`, `weathersit`, and `windspeed` intact.
# - Encode `mnth`, `hr` and `weekday` as factor variables.
# - Encode `temp` and `hum` with splines.
# %%
X, y, features = get_data_bike()
d = len(features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
features.maps_[1].cats = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
features.maps_[4].cats = ["Mon", "Thu", "Wed", "Thu", "Fri", "Sat", "Sun"]

# %%
encoder = ColumnTransformer([('identify', FunctionTransformer(), [0, 3, 5, 6, 9]),
                             ('ohe', OneHotEncoder(sparse_output=False), [1, 2, 4]),
                             ('splines', SplineTransformer(n_knots=4, degree=3, include_bias=False), [7, 8])
                            ])

model = Pipeline([('encoder', encoder), ('scaler', StandardScaler()), ('predictor', Ridge(alpha=1e-3))])
model

# %%
grid = {"predictor__alpha": np.logspace(-4, 4, 40),
        'encoder__splines__n_knots' : [3, 4, 5, 6],
        'encoder__splines__degree' : [1, 2, 3]}

search = GridSearchCV(
    model,
    cv=KFold(),
    scoring='neg_root_mean_squared_error',
    param_grid=grid,
    verbose=2
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
plt.scatter(res['param_predictor__alpha'], cv_perf, c='b', alpha=0.75)
plt.plot(res['param_predictor__alpha'][best_idx], cv_perf[best_idx], 'r*', markersize=10, markeredgecolor='k')
plt.xlabel("Regularisation alpha")
plt.ylabel("Cross-Validated RMSE")
plt.xscale('log')
plt.ylim(97, 120)
plt.show()

# %%

# Pickle the model
dump(model, os.path.join("models", "bike_splines.joblib"))

# %% [markdown]
# Here we load the model. On subsequent reruns of the script,
# you do not need to retrain the model and can simply start by running
# this cell.
# %%
# Load the model
model = load(os.path.join("models", "bike_splines.joblib"))
print(model.score(X_test, y_test))

# %%[markdown]
## Local Feature Attributions
# Here we compute local feature attributions of the
# additive model. These attributions are then visualized as
# **Partial Dependence Plots**.
# %%

foreground = X
background = X
components = get_components_linear(model, foreground, background)

# %%
partial_dependence_plot(components, foreground, background, features, Imap_inv=features.Imap_inv, plot_hist=True, 
                        n_cols=5, figsize=(30, 15))
plt.show()

# %% [markdown]
# We next show the feature attributions of randomly chosen
# examples from the dataset
# %%

for idx in range(0, 20, 2):
    print(f"##### Instance {idx} #####")
    # Choose an instance
    x_explain = X[[idx]]
    x_map = features.print_value(x_explain.ravel())

    # Predictions and explanation
    pred = model.predict(x_explain)
    bar(np.array([components[(i,)][idx] for i in range(d)]), x_map)
    plt.title(f"Target {y[idx]} Prediction {pred[0]:.3f}")
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
features_names = features.print_names()
bar(FI, features_names)
plt.show()
top_k = np.argsort(-FI)[:5]
print(f"Top 5 features {[features_names[i] for i in top_k]}")

# %%
