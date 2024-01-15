""" Assert that TreeSHAP returns the right values """

import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from scipy.stats import chi2
import time

import shap
from shap.explainers import Tree, Exact
from shap.maskers import Independent

from pyfd.decompositions import interventional_treeshap, get_components_brute_force
from pyfd.decompositions import additive_treeshap, additive_leafshap, taylor_treeshap



def compare_shap_implementations(X, model, black_box, sym=False):
    if X.shape[0] > 500:
        X = X[:500]
    background = X[:100]
    # Use the same foreground as background
    if sym:
        X = background

    # Run the custom treeshap (non-anchored)
    custom_shap, _ = interventional_treeshap(model, X, background, anchored=False)

    # Run the exact explainer
    masker = Independent(background, max_samples=100)
    explainer = Exact(black_box, masker=masker)
    orig_shap = explainer(X).values

    # Make sure we output the same result
    assert np.isclose(orig_shap, custom_shap).all(), "Non-Anchored Different from ExactExplainer"


    # Run the custom treeshap (anchored)
    custom_shap, _ = interventional_treeshap(model, X, background, anchored=True)

    # Make sure we output the same result
    assert np.isclose(orig_shap, custom_shap.mean(1)).all(), "Anchored Different from ExactExplainer"

    # Make sure that anchored-SHAP sum up to h(x) - h(z)
    gaps = black_box(X).reshape((-1, 1)) - black_box(background)
    assert  np.isclose(gaps, custom_shap.sum(-1)).all(), "Anchored does not sum to h(x) - h(z)"



def check_shap_additivity(X, model, black_box, Imap_inv):
    if X.shape[0] > 500:
        X = X[:500]
    background = X[:100]

    # Gap h(x) - E[ h(z) ]
    gaps = black_box(X) - black_box(background).mean()
    
    # Run the custom treeshap (not anchored)
    custom_shap, _ = interventional_treeshap(model, X, background, Imap_inv=Imap_inv)

    assert custom_shap.shape[1] == len(Imap_inv), "Not one SHAP value per coallition"

    # Make sure the SHAP values add up to the gaps
    assert np.isclose(gaps, custom_shap.sum(1)).all(), "SHAP does not sum to h(x) - E[h(z)]"


    # Run the custom treeshap (anchored)
    custom_shap, _ = interventional_treeshap(model, X, background, Imap_inv=Imap_inv, anchored=True)

    assert custom_shap.shape[2] == len(Imap_inv), "Not one SHAP value per coallition"

    # Make sure that anchored-SHAP sum up to h(x) - h(z)
    gaps = black_box(X).reshape((-1, 1)) - black_box(background)
    assert  np.isclose(gaps, custom_shap.sum(-1)).all(), "Anchored does not sum to h(x) - h(z)"



def compare_additive_implementations(X, model, black_box, sym=False, Imap_inv=None):
    if X.shape[0] > 500:
        X = X[:500]
    background = X[:100]
    # Use the same foreground as background
    if sym:
        X = background

    # Run the custom treeshap (non-anchored)
    start = time.time()
    custom_additive, _ = additive_treeshap(model, X, background, anchored=False, Imap_inv=Imap_inv)
    end = time.time()
    print(f"Custom additive interventional took {end-start:.1f} seconds")
    
    # Run the brute force method (non-anchored)
    decomp = get_components_brute_force(black_box, X, background, anchored=False, Imap_inv=Imap_inv)

    # Make sure we output the same result
    for i in range(custom_additive.shape[-1]):
        assert np.isclose(custom_additive[..., i], decomp[(i,)][:, 0]).all(), "Non-Anchored Different from brute force"
    

    # Run the custom treeshap (anchored)
    start = time.time()
    custom_additive, _ = additive_treeshap(model, X, background, anchored=True, Imap_inv=Imap_inv)
    end = time.time()
    print(f"Custom additive anchored took {end-start:.1f} seconds")

    # Run the brute force method (anchored)
    decomp = get_components_brute_force(black_box, X, background, anchored=True, Imap_inv=Imap_inv)

    # Make sure we output the same result
    for i in range(custom_additive.shape[-1]):
        assert np.isclose(custom_additive[..., i], decomp[(i,)]).all(), "Anchored Different from brute force"


def compare_shap_taylor_implementations(X, model, black_box):
    if X.shape[0] > 50:
        X = X[:50]
    # Run the original treeshap
    masker = Independent(X, max_samples=50)
    explainer = Exact(black_box, masker=masker)
    orig_shap_taylor = explainer(X, interactions=2).values

    # Run the custom treeshap
    custom_shap_taylor, _ = taylor_treeshap(model, X, X)

    # Make sure we output the same result
    assert np.isclose(orig_shap_taylor, custom_shap_taylor).all()

    # Make sure that shap_taylor sums up to shap
    custom_shap, _ = interventional_treeshap(model, X, X)
    assert np.isclose(custom_shap_taylor.sum(-1), custom_shap).all()



def compare_leafshap_treeshap(X, model, Imap_inv=None):
    if X.shape[0] > 500:
        X = X[:500]
    background = X[:100]

    treeshap, _ = additive_treeshap(model, X, background, Imap_inv=Imap_inv)
    leafshap, _ = additive_leafshap(model, X, background, Imap_inv=Imap_inv)

    # Make sure we output the same result
    assert np.isclose(treeshap, leafshap).all(), "TreeSHAP and LeafSHAP disagree"

    # TODO SHAP values



def setup_task(d, correlations, model_name, task):
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


#############################################################################################################
#                                                  Tests                                                    #
#############################################################################################################


####### TreeSHAP on toy data ########
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("d", [4, 8])
@pytest.mark.parametrize("correlations", [False, True])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("model_name", ["rf", "gbt", "extra", "hist"])
def test_toy_implementation(sym, d, correlations, task, model_name):

    # Setup data and model
    X, y, model, black_box = setup_task(d, correlations, model_name, task)

    # Compare Additive with the brute-force approach
    compare_additive_implementations(X, model, black_box, sym)

    # Compute SHAP values
    compare_shap_implementations(X, model, black_box, sym)

    # Compute Shapley-Taylor values
    compare_shap_taylor_implementations(X, model, black_box)

    # Compare leafshap and treeshap
    compare_leafshap_treeshap(X, model)



####### TreeSHAP on toy data with grouping ########
@pytest.mark.parametrize("d", range(4, 21, 4))
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("model_name", ["rf", "gbt", "extra", "hist"])
def test_toy_coallition(d, task, model_name):
    np.random.seed(42)

    # Setup data and model
    X, y, model, black_box = setup_task(d, False, model_name, task)

    # Determine coallitions
    n_coallitions = d//4
    Imap_inv = [list(range(i*4, (i+1)*4)) for i in range(n_coallitions)]

    # Compute SHAP values
    check_shap_additivity(X, model, black_box, Imap_inv)

    # Compare additive with the brute-force approach
    compare_additive_implementations(X, model, black_box, sym=False, Imap_inv=Imap_inv)

    # Compare leafshap and treeshap
    compare_leafshap_treeshap(X, model, Imap_inv)



# Test with adult with/without one-hot-encoding and grouping
@pytest.mark.parametrize("with_ohe", [False, True])
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_adult(with_ohe, model_name, grouping):

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
    
    # Compute SHAP values
    if grouping:
        check_shap_additivity(X, model, black_box, Imap_inv)
    else:
        compare_shap_implementations(X, model, black_box)

    # Compare additive decompositions
    compare_additive_implementations(X, model, black_box, Imap_inv=Imap_inv)



# Test with bike with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_bike(model_name, grouping):

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
    
    # Compute SHAP values
    if grouping:
        check_shap_additivity(X, model, black_box, Imap_inv)
    else:
        compare_shap_implementations(X, model, black_box)

    # Compare additive decompositions
    compare_additive_implementations(X, model, black_box, Imap_inv=Imap_inv)

    # Compare SHAP-Taylor values
    if not grouping:
        compare_shap_taylor_implementations(X, model, black_box)



# Test with california with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_california(model_name, grouping):

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
    
    # Compute SHAP values
    if grouping:
        check_shap_additivity(X, model, black_box, Imap_inv)
    else:
        compare_shap_implementations(X, model, black_box)

    # Compare additive decompositions
    compare_additive_implementations(X, model, black_box, Imap_inv=Imap_inv)

    # Compare SHAP-Taylor values
    if not grouping:
        compare_shap_taylor_implementations(X, model, black_box)





# Test with bike with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_compas(model_name, grouping):

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
    
    # Compute SHAP values
    if grouping:
        check_shap_additivity(X, model, black_box, Imap_inv)
    else:
        compare_shap_implementations(X, model, black_box)

    # Compare additive decompositions
    compare_additive_implementations(X, model, black_box, Imap_inv=Imap_inv)

    # Compare SHAP-Taylor values
    if not grouping:
        compare_shap_taylor_implementations(X, model, black_box)



if __name__ == "__main__":
    test_toy_implementation(4, False, "regression", "gbt")
    # test_adult(False, "xgb", True)
    # test_compas("rf", False)
    # test_california("gbt", True)

