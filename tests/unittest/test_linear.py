""" Assert that Additive models attributions are exact """

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import SGDRegressor, SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, SplineTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from scipy.stats import chi2

from pyfd.decompositions import get_components_linear, get_components_brute_force
from pyfd.features import Features


def setup_toy_task(d_num, d_cat=0, n_samples=500, task="regression"):
    np.random.seed(42)
    # Generate input
    d = d_num + d_cat
    X_num = np.random.normal(0, 1, size=(n_samples, d_num))
    if d_cat > 0:
        X_cat = np.random.randint(0, 5, size=(n_samples, d_cat))
        X = np.hstack((X_num, X_cat))
    else:
        X_cat = 0
        X = X_num
    features = Features(X, [f"x{i}" for i in range(d)], ["num"]*d)
    # Generate target
    if task == "regression":
        y = ((0.5+X)**2).mean(1)
    else:
        threshold = np.sqrt(chi2(df=d).ppf(0.5))
        y = (np.linalg.norm(X_num, axis=1) > threshold).astype(int)
        
    return X, y.ravel(), features




####### Linear Models on toy data ########
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("num_encoding", ["identity", "bins", "splines-bias", "splines-nbias"])
@pytest.mark.parametrize("drop_ohe_cat", [True, False])
def test_toy_linear(task, num_encoding, drop_ohe_cat):

    # Setup data and model
    d_num = 2
    d_cat = 2
    d = d_num + d_cat
    X, y, features = setup_toy_task(d_num, d_cat, 300, task)
    # Encoding numerical features
    if num_encoding == "identity": 
        numerical_encoder = FunctionTransformer()
    elif num_encoding == "bins": 
        if drop_ohe_cat:
            numerical_encoder = Pipeline([('discretizer', KBinsDiscretizer(encode='ordinal')),
                                          ('ohe', OneHotEncoder(drop='first'))])
        else:
            numerical_encoder = KBinsDiscretizer(encode='onehot-dense')
    elif num_encoding == "splines-bias": 
        numerical_encoder = SplineTransformer(n_knots=4, knots='quantile', include_bias=True)
    elif num_encoding == "splines-nbias": 
        numerical_encoder = SplineTransformer(n_knots=4, knots='quantile', include_bias=False)
    # Encoding categorical features
    ohe_encoder = OneHotEncoder(sparse_output=False, drop='first') if drop_ohe_cat else\
                  OneHotEncoder(sparse_output=False)
    encoder = ColumnTransformer([
                ('num', numerical_encoder, list(range(d_num))), ('cat', ohe_encoder, list(range(d_num, d)) )
                ])
    if task == "regression":
        lin_models = [Ridge(), Lasso(), ElasticNet(), SGDRegressor()]
    else:
        lin_models = [LogisticRegression(), SGDClassifier(), LinearSVC()]

    # Try out multiple models
    for lin_model in lin_models:
        model = Pipeline([('encoder', encoder), ('scaler', StandardScaler()), ('predictor', lin_model)])
        model.fit(X, y)

        # Explain the model
        foreground = X
        background = X
        components = get_components_linear(model, foreground, background, features)

        # Sanity check
        if task == "regression":
            h = model.predict
        else:
            h = model.decision_function
        components_2 = get_components_brute_force(h, foreground, background, features)
        for i in range(d):
            assert np.isclose(components[(i,)], components_2[(i,)].mean(1)).all(), f"Fail for {type(lin_model)}"



if __name__ == "__main__":
    test_toy_linear("regression", "identity", False)

