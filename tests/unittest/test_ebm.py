""" Assert that Additive models attributions are exact """

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from scipy.stats import chi2

from pyfd.decompositions import get_components_ebm, get_components_brute_force



def setup_toy_task_additive(d_num, d_cat=0, n_samples=500, task="regression"):
    np.random.seed(42)
    # Generate input
    d = d_num + d_cat
    X_num = np.random.normal(0, 1, size=(n_samples, d_num))
    if d_cat > 0:
        X_cat = np.random.randint(0, 10, size=(n_samples, d_cat))
        X = np.hstack((X_num, X_cat))
    else:
        X_cat = 0
        X = X_num
    # Generate target
    if task == "regression":
        y = ((0.5+X)**2).mean(1)
    else:
        threshold = np.sqrt(chi2(df=d).ppf(0.5))
        y = (np.linalg.norm(X_num, axis=1) > threshold).astype(int)
        
    return X, y.ravel()




# def setup_toy_task_interactions(d_num, d_cat=0, n_samples=500, task="regression"):
#     np.random.seed(42)
#     # Generate input
#     d = d_num + d_cat
#     X_num = np.random.normal(0, 1, size=(n_samples, d_num))
#     if d_cat > 0:
#         X_cat = np.random.randint(0, 10, size=(n_samples, d_cat))
#         X = np.hstack((X_num, X_cat))
#     else:
#         X_cat = 0
#         X = X_num
#     # Generate target
#     if task == "regression":
#         y = ((0.5+X)**2).mean(1)
#     else:
#         threshold = np.sqrt(chi2(df=d).ppf(0.5))
#         y = (np.linalg.norm(X_num, axis=1) > threshold).astype(int)

#     return X, y.ravel()



####### Linear Models on toy data ########
@pytest.mark.parametrize("d_num", [5, 10])
@pytest.mark.parametrize("d_cat", [0, 5])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("num_encoding", ["identity", "standard", "min-max"])
def test_toy_ebm_additive(d_num, d_cat, task, num_encoding):

    # Setup data and model
    d = d_num + d_cat
    X, y = setup_toy_task_additive(d_num, d_cat, 1000, task)
    numerical_encoders = {"identity": FunctionTransformer(),
                          "standard": StandardScaler(),
                          "min-max":  MinMaxScaler()}
    if task == "regression":
        model = ExplainableBoostingRegressor(interactions=0)
    else:
        model = ExplainableBoostingClassifier(interactions=0)
    if d_cat > 0:
        encoder = ColumnTransformer([
                    ('num', numerical_encoders[num_encoding], list(range(d_num))),
                    ('cat', "passthrough", list(range(d_num, d)))
                    ])
    else:
        encoder = numerical_encoders[num_encoding]
    model = Pipeline([('encoder', encoder), ('predictor', model)])
    model.fit(X, y)

    # Explain the model
    foreground = X
    background = X
    components = get_components_ebm(model, foreground, background)

    # Sanity check
    if task == "regression":
        h = model.predict
    else:
        h = model.decision_function
    components_2 = get_components_brute_force(h, foreground, background)
    for i in range(d):
        assert np.isclose(components[(i,)], components_2[(i,)].mean(1)).all()



if __name__ == "__main__":
    test_toy_ebm_additive(3, 2, "regression", "identity")

