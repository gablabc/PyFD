""" Assert that TreeSHAP returns the right values """

import pytest
import numpy as np
from shap.explainers import Exact
from shap.maskers import Independent

from pyfd.shapley import interventional_treeshap, taylor_treeshap

from utils import setup_toy, setup_adult, setup_bike, setup_california, setup_compas

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
    X, y, model, black_box = setup_toy(d, correlations, model_name, task)

    # Compute SHAP values
    compare_shap_implementations(X, model, black_box, sym)

    # Compute Shapley-Taylor values
    compare_shap_taylor_implementations(X, model, black_box)




####### TreeSHAP on toy data with grouping ########
@pytest.mark.parametrize("d", range(4, 21, 4))
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("model_name", ["rf", "gbt", "extra", "hist"])
def test_toy_coallition(d, task, model_name):
    np.random.seed(42)

    # Setup data and model
    X, y, model, black_box = setup_toy(d, False, model_name, task)

    # Determine coallitions
    n_coallitions = d//4
    Imap_inv = [list(range(i*4, (i+1)*4)) for i in range(n_coallitions)]

    # Compute SHAP values
    check_shap_additivity(X, model, black_box, Imap_inv)



# Test with adult with/without one-hot-encoding and grouping
@pytest.mark.parametrize("with_ohe", [False, True])
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_adult(with_ohe, model_name, grouping):

    X, model, black_box, Imap_inv = setup_adult(with_ohe, model_name, grouping)
    
    # Compute SHAP values
    if grouping:
        check_shap_additivity(X, model, black_box, Imap_inv)
    else:
        compare_shap_implementations(X, model, black_box)



# Test with bike with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_bike(model_name, grouping):

    X, model, black_box, Imap_inv = setup_bike(model_name, grouping)
    
    # Compute SHAP values
    if grouping:
        check_shap_additivity(X, model, black_box, Imap_inv)
    else:
        compare_shap_implementations(X, model, black_box)

    # Compare SHAP-Taylor values
    if not grouping:
        compare_shap_taylor_implementations(X, model, black_box)



# Test with california with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_california(model_name, grouping):

    X, model, black_box, Imap_inv = setup_california(model_name, grouping)
    
    # Compute SHAP values
    if grouping:
        check_shap_additivity(X, model, black_box, Imap_inv)
    else:
        compare_shap_implementations(X, model, black_box)

    # Compare SHAP-Taylor values
    if not grouping:
        compare_shap_taylor_implementations(X, model, black_box)





# Test with bike with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_compas(model_name, grouping):

    X, model, black_box, Imap_inv = setup_compas(model_name, grouping)
    
    # Compute SHAP values
    if grouping:
        check_shap_additivity(X, model, black_box, Imap_inv)
    else:
        compare_shap_implementations(X, model, black_box)

    # Compare SHAP-Taylor values
    if not grouping:
        compare_shap_taylor_implementations(X, model, black_box)



if __name__ == "__main__":
    test_toy_implementation(4, False, "regression", "gbt")
    # test_adult(False, "xgb", True)
    # test_compas("rf", False)
    # test_california("gbt", True)

