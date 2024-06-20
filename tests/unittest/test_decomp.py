""" Assert that the Decomposition Functions are correct """

import pytest
import numpy as np
import shap
from shap.maskers import Independent

from pyfd.decompositions import get_components_brute_force, get_components_adaptive
from pyfd.shapley import lattice_shap


def generate_problem(N):
    # Generate the data
    np.random.seed(42)
    X = np.random.uniform(-1, 1, size=(N, 5))
    def h(X):
        return X[:, 0] + X[:, 1] + 5 * X[:, 0] * X[:, 1] + 0.5 * np.sin(2*np.pi*X[:, 2])
    return X, h



def compare_adaptive_classical_SHAP(h, U, X):
    # Adaptive Interventional Decomposition and SHAP values
    h_components = get_components_adaptive(h, X, tolerance=1e-5)
    U_ = list(h_components.keys())
    assert len(U) == len(U_)
    assert np.array([u in U_ for u in U]).all()
    shap_values = lattice_shap(h, X, X, interactions=U_)

    # Classical SHAP values
    masker = Independent(X, max_samples=X.shape[0])
    explainer = shap.explainers.Exact(h, masker)
    shap_values_original = explainer(X).values

    assert np.isclose(shap_values, shap_values_original).all()


#############################################################################################################
#                                                  Tests                                                    #
#############################################################################################################


def test_brute_force_IO():
    """ Test the consistency of the output type depending on various inputs """
    N = 500
    X, h = generate_problem(N)

    # test ICE
    x_lin = np.linspace(-1, 1, 100)
    for i in range(5):
        decomp = get_components_brute_force(h, x_lin, X, Imap_inv=[[i]])
        assert len(decomp.keys()) == 2, "ICE failed"
        assert decomp[()].shape == (N,), "ICE failed"
        assert decomp[(0,)].shape == (100, N), "ICE failed"

    # test PDP
    for i in range(5):
        decomp = get_components_brute_force(h, x_lin, X, Imap_inv=[[i]], anchored=False)
        assert len(decomp.keys()) == 2, "PDP failed"
        assert decomp[()].shape == (N,), "PDP failed"
        assert decomp[(0,)].shape == (100,), "PDP failed"

    # test up to pair-wise interactions among all features
    decomp = get_components_brute_force(h, X, X, anchored=True, interactions=2)
    assert len(decomp.keys()) == 16, "Pairwise interaction failed"
    assert decomp[()].shape == (N,), "Pairwise interaction failed"
    assert decomp[(0,)].shape == (N, N), "Pairwise interaction failed"
    assert decomp[(0, 1)].shape == (N, N), "Pairwise interaction failed"

    # test three-way interactions among all features
    decomp = get_components_brute_force(h, X, X, anchored=True, interactions=3)
    assert len(decomp.keys()) == 26, "Three-way interaction failed"
    assert decomp[()].shape == (N,), "Three-way interaction failed"
    assert decomp[(0,)].shape == (N, N), "Three-way interaction failed"
    assert decomp[(0, 1)].shape == (N, N), "Three-way interaction failed"
    assert decomp[(0, 1, 2)].shape == (N, N), "Three-way interaction failed"

    # test custom interactions
    decomp = get_components_brute_force(h, X, X, anchored=True, interactions=[(0,), (1,), (2,), (0, 1), (0, 2)])
    assert len(decomp.keys()) == 6, "Custom interaction failed"
    decomp = get_components_brute_force(h, X, X, anchored=True, interactions=[(0,), (1,), (0, 1), (2,), (0, 2)])
    assert len(decomp.keys()) == 6, "Custom interaction with unordered interactions failed"

    # test PDP with grouped features
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    decomp = get_components_brute_force(h, grid, X, Imap_inv=[[0, 1]], anchored=False)
    assert len(decomp.keys()) == 2, "Grouped PDP failed"
    assert decomp[()].shape == (N,), "Grouped PDP failed"
    assert decomp[(0,)].shape == (100,), "Grouped PDP failed"

    # Grouped features
    decomp = get_components_brute_force(h, X, X, anchored=True, Imap_inv=[[0, 1], [2, 3]], interactions=2)
    assert len(decomp.keys()) == 4, "Grouped interaction failed"
    assert decomp[()].shape == (N,), "Grouped interaction failed"
    assert decomp[(0,)].shape == (N, N), "Grouped interaction failed"
    assert decomp[(0, 1)].shape == (N, N), "Grouped interaction failed"

    # Grouped features custom interactions
    decomp = get_components_brute_force(h, X, X, anchored=True, Imap_inv=[[0, 1], [2, 3], [4]], 
                                                                interactions=[(0,), (1,), (2,), (0, 1), (0, 2)])
    assert len(decomp.keys()) == 6, "Grouped custom interaction failed"
    assert decomp[()].shape == (N,), "Grouped custom interaction failed"
    assert decomp[(0,)].shape == (N, N), "Grouped custom interaction failed"
    assert decomp[(0, 1)].shape == (N, N), "Grouped custom interaction failed"



def test_brute_force_exceptions():
    """ Assert that exceptions are correctly raised """
    N = 500
    X, h = generate_problem(N)
    
    # Passing a x_lin when inapropriate
    x_lin = np.linspace(-1, 1, 100)
    with pytest.raises(Exception):
        get_components_brute_force(h, x_lin, X, anchored=True, interactions=1)

    # Invalid interaction
    with pytest.raises(Exception):
        get_components_brute_force(h, X, X, anchored=True, interactions=10)
    
    # test custom interactions Exception
    with pytest.raises(Exception):
        get_components_brute_force(h, X, X, anchored=True, interactions=[(0,), (1,), (2,), (0, 1), (0, 1, 2)])



def test_adaptive():
    """ Assert that adaptive decompositions return the ground truth """

    # Generate the data
    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(1000, 8))

    def h(X):
        return X[:, 0] * X[:, 1] + 2 * X[:, 2] * X[:, 3]
    U = [(), (0,), (1,), (0, 1), (2,), (3,), (2, 3), (4,), (5,), (6,), (7,)]
    compare_adaptive_classical_SHAP(h, U, X)

    def h(X):
        return X[:, 1] * X[:, 2] * X[:, 3]
    U = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,),
        (1, 2), (2, 3), (1, 3), (1, 2, 3)]
    compare_adaptive_classical_SHAP(h, U, X)

    def h(X):
        return X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]
    U = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,),
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)]
    compare_adaptive_classical_SHAP(h, U, X)



if __name__ == "__main__":
    test_brute_force_IO()