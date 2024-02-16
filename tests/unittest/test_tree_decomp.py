""" Test funcitonal decompositions of Trees ensembles """

import pytest
import numpy as np

from pyfd.decompositions import get_components_brute_force, get_components_tree

from utils import setup_toy, setup_adult, setup_bike, setup_california, setup_compas


def compare_recursive_brute(X, model, black_box, sym=False, Imap_inv=None):
    if X.shape[0] > 500:
        X = X[:500]
    background = X[:100]
    # Use the same foreground as background
    if sym:
        X = background

    # Run the tree decomp (non-anchored)
    tree_decomp = get_components_tree(model, X, background, anchored=False, Imap_inv=Imap_inv)
    
    # Run the brute force method (non-anchored)
    brute_decomp = get_components_brute_force(black_box, X, background, anchored=False, Imap_inv=Imap_inv)

    # Make sure we output the same result
    for key in tree_decomp.keys():
        assert np.isclose(tree_decomp[key], brute_decomp[key]).all(), "Non-Anchored tree from brute force"
    

    # Run the tree decomp (anchored)
    tree_decomp = get_components_tree(model, X, background, anchored=True, Imap_inv=Imap_inv)

    # Run the brute force method (anchored)
    brute_decomp = get_components_brute_force(black_box, X, background, anchored=True, Imap_inv=Imap_inv)

    # Make sure we output the same result
    for key in tree_decomp.keys():
        assert np.isclose(tree_decomp[key], brute_decomp[key]).all(), "Anchored tree from brute force"



def compare_recursive_leaf(X, model, Imap_inv=None):
    if X.shape[0] > 500:
        X = X[:500]
    background = X[:100]

    recurse_decomp = get_components_tree(model, X, background, Imap_inv=Imap_inv)
    leaf_decomp = get_components_tree(model, X, background, Imap_inv=Imap_inv, algorithm="leaf")

    # Make sure we output the same result
    for key in recurse_decomp.keys():
        assert np.isclose(recurse_decomp[key][..., 0], leaf_decomp[key][..., 0]).all(), "Tree recursive and Tree leaf disagree"



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

    # Compare Additive with the brute-force approach
    compare_recursive_brute(X, model, black_box, sym)

    # Compare leafshap and treeshap
    compare_recursive_leaf(X, model)



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

    # Compare additive with the brute-force approach
    compare_recursive_brute(X, model, black_box, sym=False, Imap_inv=Imap_inv)

    # Compare leafshap and treeshap
    compare_recursive_leaf(X, model, Imap_inv)



# Test with adult with/without one-hot-encoding and grouping
@pytest.mark.parametrize("with_ohe", [False, True])
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_adult(with_ohe, model_name, grouping):

    X, model, black_box, Imap_inv = setup_adult(with_ohe, model_name, grouping)

    # Compare additive decompositions
    compare_recursive_brute(X, model, black_box, Imap_inv=Imap_inv)



# Test with bike with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_bike(model_name, grouping):

    X, model, black_box, Imap_inv = setup_bike(model_name, grouping)

    # Compare additive decompositions
    compare_recursive_brute(X, model, black_box, Imap_inv=Imap_inv)



# Test with california with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_california(model_name, grouping):

    X, model, black_box, Imap_inv = setup_california(model_name, grouping)

    # Compare additive decompositions
    compare_recursive_brute(X, model, black_box, Imap_inv=Imap_inv)





# Test with bike with/without grouping
@pytest.mark.parametrize("model_name", ["rf", "gbt", "xgb"])
@pytest.mark.parametrize("grouping", [False, True])
def test_compas(model_name, grouping):

    X, model, black_box, Imap_inv = setup_compas(model_name, grouping)

    # Compare additive decompositions
    compare_recursive_brute(X, model, black_box, Imap_inv=Imap_inv)



if __name__ == "__main__":
    test_toy_implementation(False, 4, False, "classification", "rf")
    

