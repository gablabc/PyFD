""" Utility functions for unit test """

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from scipy.stats import chi2



def setup_toy(d, correlations, model_name, task):
    np.random.seed(42)
    # Generate input
    if correlations:
        mu = np.zeros(d)
        sigma = 0.5 * np.eye(d) + 0.5 * np.ones((d, d))
        X = np.random.multivariate_normal(mean=mu, cov=sigma, size=(1000,))
    else:
        X = np.random.normal(0, 1, size=(1000, d))
    
    # from sklearn.preprocessing import StandardScaler
    # X = StandardScaler().fit_transform(X)

    # Generate target and fit model
    if task == "regression":
        y = X.mean(1)
        if model_name == "rf":
            model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
        elif model_name == "gbt":
            model = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42).fit(X, y)
        elif model_name == "extra":
            model = ExtraTreesRegressor(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
        elif model_name == "hist":
            model = HistGradientBoostingRegressor(max_iter=20, max_depth=5, random_state=42).fit(X, y)
        else:
            model = XGBRegressor(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
    else:
        y = (np.linalg.norm(X, axis=1) > np.sqrt(chi2(df=d).ppf(0.5))).astype(int)
        if model_name == "rf":
            model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
        elif model_name == "gbt":
            model = GradientBoostingClassifier(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
        elif model_name == "extra":
            model = ExtraTreesClassifier(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
        elif model_name == "hist":
            model = HistGradientBoostingClassifier(max_iter=20, max_depth=5, random_state=42).fit(X, y)
        else:
            model = XGBClassifier(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
    
    if task == "regression":
        black_box = model.predict
    else:
        if model_name in ["rf", "extra"]:
            black_box = lambda x : model.predict_proba(x)[:, -1]
        elif model_name in ["gbt", "hist"]:
            black_box = model.decision_function
        else:
            black_box = lambda x :model.predict(x, output_margin=True)
    return X, y, model, black_box




def setup_adult(with_ohe, model_name, grouping):

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

    from pyfd.data import get_data_adults

    X, y, features  = get_data_adults(use_target_encoder=not with_ohe)

    # Some arbitrary feature groups
    if grouping:
        Imap_inv = [[0], [1], [2, 3, 4], [5], [6], [7, 8, 9], [10, 11]]
    # Each feature is its own group
    else:
        Imap_inv = [[i] for i in range(X.shape[1])]
    if with_ohe:
        ohe = ColumnTransformer([
                                ('id', FunctionTransformer(), features.ordinal),
                                ('ohe', OneHotEncoder(sparse=False), features.nominal)])
    else:
        ohe = FunctionTransformer()

    # Fit model
    if model_name == "rf":
        model = Pipeline([('ohe', ohe), 
                          ('predictor', RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42))])
        model.fit(X, y)
        black_box = lambda x : model.predict_proba(x)[:, -1]
    elif model_name == "gbt":
        model = Pipeline([('ohe', ohe), 
                          ('predictor', GradientBoostingClassifier(n_estimators=20, max_depth=5, random_state=42))])
        model.fit(X, y)
        black_box = model.decision_function
    else:
        model = Pipeline([('ohe', ohe),
                          ('predictor', XGBClassifier(n_estimators=20, max_depth=5, random_state=42))])
        model.fit(X, y)
        black_box = lambda x :model.predict(x, output_margin=True).astype(np.float64)
    
    return X, model, black_box, Imap_inv



def setup_bike(model_name, grouping):

    from pyfd.data import get_data_bike

    X, y, _  = get_data_bike()

    # Some arbitrary feature groups
    if grouping:
        Imap_inv = [[0], [1], [2], [3, 4, 5, 6, 7], [8, 9]]
    # Each feature is its own group
    else:
        Imap_inv = [[i] for i in range(X.shape[1])]

    # Fit model
    if model_name == "rf":
        model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
    elif model_name == "gbt":
        model = GradientBoostingRegressor(n_estimators=20, max_depth=5, random_state=42)
    else:
        model = XGBRegressor(n_estimators=20, max_depth=2, random_state=42)
    model.fit(X, y)
    black_box = lambda x : model.predict(x)
    
    return X, model, black_box, Imap_inv



def setup_california(model_name, grouping):

    from pyfd.data import get_data_california_housing

    X, y, _  = get_data_california_housing()

    # Some arbitrary feature groups
    if grouping:
        Imap_inv = [[0], [1], [2], [3, 4, 5], [6, 7]]
    # Each feature is its own group
    else:
        Imap_inv = [[i] for i in range(X.shape[1])]

    # Fit model
    if model_name == "rf":
        model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
    elif model_name == "gbt":
        model = GradientBoostingRegressor(n_estimators=20, max_depth=5, random_state=42)
    else:
        model = XGBRegressor(n_estimators=20, max_depth=2, random_state=42)
    model.fit(X, y)
    black_box = lambda x : model.predict(x)
    
    return X, model, black_box, Imap_inv



def setup_compas(model_name, grouping):

    from pyfd.data import get_data_compas

    X, y, _  = get_data_compas()

    # Some arbitrary feature groups
    if grouping:
        Imap_inv = [[0, 1, 2], [3], [4], [5, 6]]
    # Each feature is its own group
    else:
        Imap_inv = [[i] for i in range(X.shape[1])]

    # Fit model
    if model_name == "rf":
        model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
        model.fit(X, y)
        black_box = lambda x : model.predict_proba(x)[:, -1]
    elif model_name == "gbt":
        model = GradientBoostingClassifier(n_estimators=20, max_depth=5, random_state=42)
        model.fit(X, y)
        black_box = model.decision_function
    else:
        model = XGBClassifier(n_estimators=20, max_depth=5, random_state=42)
        model.fit(X, y)
        black_box = lambda x :model.predict(x, output_margin=True)
    
    return X, model, black_box, Imap_inv
