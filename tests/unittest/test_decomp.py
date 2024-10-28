""" Assert that the Decomposition Functions are correct """

import pytest
import numpy as np
import shap
from shap.maskers import Independent

from pyfd.decompositions import get_components_brute_force, get_components_adaptive
from pyfd.decompositions import get_CoE, get_interventional_from_anchored
from pyfd.decompositions import get_regional_decompositions
from pyfd.shapley import lattice_shap
from pyfd.features import Features


def generate_problem(N, discrete=False):
    # Generate the data
    np.random.seed(42)
    if discrete:
        X = np.random.randint(0, 5, size=(N, 5))
    else:
        X = np.random.uniform(-1, 1, size=(N, 5))
    def h(X):
        return X[:, 0] + X[:, 1] + 5 * X[:, 0] * X[:, 1] + 0.5 * np.sin(2*np.pi*X[:, 2])
    features = Features(X, [f"x{i}" for i in range(5)], ["num"]*5)
    return X, h, features



def compare_adaptive_classical_SHAP(h, U, X, features):
    # Adaptive Interventional Decomposition and SHAP values
    h_components = get_components_adaptive(h, X, features, tolerance=1e-5)
    U_ = list(h_components.keys())
    assert len(U) == len(U_)
    assert np.array([u in U_ for u in U]).all()
    shap_values = lattice_shap(h, X, X, features, interactions=U_)

    # Classical SHAP values
    masker = Independent(X, max_samples=X.shape[0])
    explainer = shap.explainers.Exact(h, masker)
    shap_values_original = explainer(X).values

    assert np.isclose(shap_values, shap_values_original).all()


#############################################################################################################
#                                                  Tests                                                    #
#############################################################################################################


@pytest.mark.parametrize("discrete", [False, True])
def test_brute_force_IO(discrete):
    """ Test the consistency of the output type depending on various inputs """
    N = 500
    X, h, features = generate_problem(N, discrete)

    # test ICE
    x_lin = np.linspace(-1, 1, 100)
    for i in range(5):
        decomp = get_components_brute_force(h, x_lin, X, features.select([i]))
        assert len(decomp.keys()) == 2, "ICE failed"
        assert decomp[()].shape == (N,), "ICE failed"
        assert decomp[(0,)].shape == (100, N), "ICE failed"

    # test PDP
    for i in range(5):
        decomp = get_components_brute_force(h, x_lin, X, features.select([i]), anchored=False)
        assert len(decomp.keys()) == 2, "PDP failed"
        assert decomp[()].shape == (N,), "PDP failed"
        assert decomp[(0,)].shape == (100,), "PDP failed"

    # test up to pair-wise interactions among all features
    decomp = get_components_brute_force(h, X, X, features, anchored=True, interactions=2)
    assert len(decomp.keys()) == 16, "Pairwise interaction failed"
    assert decomp[()].shape == (N,), "Pairwise interaction failed"
    assert decomp[(0,)].shape == (N, N), "Pairwise interaction failed"
    assert decomp[(0, 1)].shape == (N, N), "Pairwise interaction failed"

    # test three-way interactions among all features
    decomp = get_components_brute_force(h, X, X, features, anchored=True, interactions=3)
    assert len(decomp.keys()) == 26, "Three-way interaction failed"
    assert decomp[()].shape == (N,), "Three-way interaction failed"
    assert decomp[(0,)].shape == (N, N), "Three-way interaction failed"
    assert decomp[(0, 1)].shape == (N, N), "Three-way interaction failed"
    assert decomp[(0, 1, 2)].shape == (N, N), "Three-way interaction failed"

    # test custom interactions
    decomp = get_components_brute_force(h, X, X, features, anchored=True, interactions=[(0,), (1,), (2,), (0, 1), (0, 2)])
    assert len(decomp.keys()) == 6, "Custom interaction failed"
    decomp = get_components_brute_force(h, X, X, features, anchored=True, interactions=[(0,), (1,), (0, 1), (2,), (0, 2)])
    assert len(decomp.keys()) == 6, "Custom interaction with unordered interactions failed"

    # test PDP with grouped features
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    select_features = features.select([0, 1]).group([[0, 1]])
    decomp = get_components_brute_force(h, foreground=grid, background=X, features=select_features, anchored=False)
    assert len(decomp.keys()) == 2, "Grouped PDP failed"
    assert decomp[()].shape == (N,), "Grouped PDP failed"
    assert decomp[(0,)].shape == (100,), "Grouped PDP failed"

    # Grouped features
    select_features = features.select([0, 1, 2, 3]).group([[0, 1], [2, 3]])
    decomp = get_components_brute_force(h, X, X, select_features, anchored=True, interactions=2)
    assert len(decomp.keys()) == 4, "Grouped interaction failed"
    assert decomp[()].shape == (N,), "Grouped interaction failed"
    assert decomp[(0,)].shape == (N, N), "Grouped interaction failed"
    assert decomp[(0, 1)].shape == (N, N), "Grouped interaction failed"

    # Grouped features custom interactions
    select_features = features.select([0, 1, 2, 3, 4]).group([[0, 1], [2, 3], [4]])
    decomp = get_components_brute_force(h, X, X, select_features, anchored=True, interactions=[(0,), (1,), (2,), (0, 1), (0, 2)])
    assert len(decomp.keys()) == 6, "Grouped custom interaction failed"
    assert decomp[()].shape == (N,), "Grouped custom interaction failed"
    assert decomp[(0,)].shape == (N, N), "Grouped custom interaction failed"
    assert decomp[(0, 1)].shape == (N, N), "Grouped custom interaction failed"



def test_brute_force_exceptions():
    """ Assert that exceptions are correctly raised """
    N = 500
    X, h, features = generate_problem(N)
    
    # Passing a x_lin when inapropriate
    x_lin = np.linspace(-1, 1, 100)
    with pytest.raises(Exception):
        get_components_brute_force(h, x_lin, X, features, anchored=True, interactions=1)

    # Invalid interaction
    with pytest.raises(Exception):
        get_components_brute_force(h, X, X, features, anchored=True, interactions=10)
    
    # test custom interactions Exception
    with pytest.raises(Exception):
        get_components_brute_force(h, X, X, features, anchored=True, interactions=[(0,), (1,), (2,), (0, 1), (0, 1, 2)])


@pytest.mark.parametrize("discrete", [False, True])
def test_adaptive(discrete):
    """ Assert that adaptive decompositions return the ground truth """

    # Generate the data
    np.random.seed(42)
    if discrete:
        X = np.random.randint(0, 5, size=(1000, 8))
    else:
        X = np.random.uniform(0, 1, size=(1000, 8))
    features = Features(X, [f"x{i}" for i in range(8)], ["num"]*8)
    def h(X):
        return X[:, 0] * X[:, 1] + 2 * X[:, 2] * X[:, 3]
    U = [(), (0,), (1,), (0, 1), (2,), (3,), (2, 3), (4,), (5,), (6,), (7,)]
    compare_adaptive_classical_SHAP(h, U, X, features)

    def h(X):
        return X[:, 1] * X[:, 2] * X[:, 3]
    U = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,),
        (1, 2), (2, 3), (1, 3), (1, 2, 3)]
    compare_adaptive_classical_SHAP(h, U, X, features)

    def h(X):
        return X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]
    U = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,),
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)]
    compare_adaptive_classical_SHAP(h, U, X, features)


@pytest.mark.parametrize("discrete", [False, True])
def test_components_utils(discrete):
    """ Test the functions applied on top of decompositions """
    N = 500
    X, h, features = generate_problem(N, discrete)
    decomp_anchored = get_components_brute_force(h, X, X, features, anchored=True)
    decomp_interv = get_interventional_from_anchored(decomp_anchored)
    assert np.isclose(decomp_anchored[()], decomp_interv[()]).all()
    assert decomp_interv[(0,)].shape == (N,)
    assert decomp_anchored.keys() == decomp_interv.keys()

    # Compute the Cost of Exclusion
    coe_anchored = get_CoE(decomp_anchored)
    coe_interv = get_CoE(decomp_interv)
    assert np.isclose(coe_anchored, coe_interv)

    # Compute the Cost of Exclusion while passing foreground_preds
    foreground_preds = h(X)
    assert np.isclose(coe_anchored, get_CoE(decomp_anchored, foreground_preds))
    assert np.isclose(coe_interv, get_CoE(decomp_interv, foreground_preds))

    # Regional CoE
    regions = (X[:, 0] > 0).astype(int)
    regional_decompositions = get_regional_decompositions(decomp_anchored, regions, regions, 2)
    print(regional_decompositions)
    coe_regional = get_CoE(regional_decompositions)
    coe_regional_ = get_CoE(regional_decompositions, [foreground_preds[regions==0], foreground_preds[regions==1]])
    assert np.isclose(coe_regional, coe_regional_)


if __name__ == "__main__":
    test_components_utils(True)
