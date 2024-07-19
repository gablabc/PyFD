""" Assert that Additive models attributions are exact """

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from scipy.stats import chi2

from pyfd.decompositions import get_components_ebm, get_components_brute_force



def setup_toy_task(d_num, d_cat=0, n_samples=1000, task="regression", 
                        num_encoding="identity", interactions=False):
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
    if interactions:
        if task == "regression":
            y = np.exp((-X**2).mean(1))
        else:
            threshold = np.sqrt(chi2(df=d).ppf(0.5))
            y = (X_num.prod(axis=1) > threshold).astype(int)
    else:
        if task == "regression":
            y = ((0.5+X)**2).mean(1)
        else:
            threshold = np.sqrt(chi2(df=d).ppf(0.5))
            y = (np.linalg.norm(X_num, axis=1) > threshold).astype(int)
    
    # EBM
    if task == "regression":
        model = ExplainableBoostingRegressor(interactions=0.95*int(interactions), outer_bags=2)
    else:
        model = ExplainableBoostingClassifier(interactions=0.95*int(interactions), outer_bags=2)

    # Feature Embeddings
    numerical_encoders = {"standard": StandardScaler(), "min-max":  MinMaxScaler()}
    if not num_encoding == "identity":
        if d_cat > 0:
            encoder = ColumnTransformer([
                        ('num', numerical_encoders[num_encoding], list(range(d_num))),
                        ('cat', "passthrough", list(range(d_num, d)))
                        ])
        else:
            encoder = numerical_encoders[num_encoding]
        model = Pipeline([('encoder', encoder), ('predictor', model)])
    
    # Train the model
    model.fit(X, y.ravel())
    # Sanity check
    if task == "regression":
        h = model.predict
    else:
        h = model.decision_function
    
    return X, model, h



####### EBMs on toy data ########
@pytest.mark.parametrize("d_num", [5, 10])
@pytest.mark.parametrize("d_cat", [0, 5])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("num_encoding", ["identity", "standard", "min-max"])
@pytest.mark.parametrize("interactions", [False, True])
@pytest.mark.parametrize("anchored", [False, True])
def test_toy_ebm_full(d_num, d_cat, task, num_encoding, interactions, anchored):

    # Setup data and model
    X, model, h = setup_toy_task(d_num, d_cat, 1000, task, num_encoding, interactions)

    # Fully explain the model
    if type(model) == Pipeline:
        U = list(model[-1].term_features_)
    else:
        U = list(model.term_features_)
    foreground = X
    background = X[:500]
    components = get_components_ebm(model, foreground, background, anchored=anchored)
    components_2 = get_components_brute_force(h, foreground, background, 
                                                interactions=U, anchored=anchored)
    assert np.isclose(components[()], components_2[()]).all(), "Intercepts are not the same"
    for u in U:
        assert np.isclose(components[u], components_2[u]).all(), "Components are not the same"



@pytest.mark.parametrize("d_num", [5, 10])
@pytest.mark.parametrize("d_cat", [0, 5])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("num_encoding", ["identity", "standard", "min-max"])
@pytest.mark.parametrize("interactions", [False, True])
@pytest.mark.parametrize("anchored", [False, True])
def test_toy_ebm_partial(d_num, d_cat, task, num_encoding, interactions, anchored):

    # Setup data and model
    X, model, h = setup_toy_task(d_num, d_cat, 1000, task, num_encoding, interactions)

    # Partially explain the model
    foreground = X
    background = X[:500]
    components = get_components_ebm(model, foreground, background, Imap_inv=[[0], [1]], anchored=anchored)
    U = list(components.keys())[1:]
    components_2 = get_components_brute_force(h, foreground, background, Imap_inv=[[0], [1]], 
                                                        interactions=U, anchored=anchored)
    assert np.isclose(components[()], components_2[()]).all(), "Intercepts are not the same"
    for u in U:
        assert np.isclose(components[u], components_2[u]).all(), "Components are not the same"

    # Do a line PDP/ICE for feature 0
    foreground = np.linspace(-2, 2, 200)
    components = get_components_ebm(model, foreground, background, Imap_inv=[[0]], anchored=anchored)
    components_2 = get_components_brute_force(h, foreground, background, Imap_inv=[[0]], anchored=anchored)
    assert np.isclose(components[()], components_2[()]).all(), "Intercepts are not the same"
    assert np.isclose(components[(0,)], components_2[(0,)]).all(), "Components are not the same"



@pytest.mark.parametrize("d_num", [5, 10])
@pytest.mark.parametrize("d_cat", [0, 5])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("num_encoding", ["identity", "standard", "min-max"])
@pytest.mark.parametrize("interactions", [False, True])
@pytest.mark.parametrize("anchored", [False, True])
def test_toy_ebm_grouping(d_num, d_cat, task, num_encoding, interactions, anchored):

    # Setup data and model
    X, model, h = setup_toy_task(d_num, d_cat, 1000, task, num_encoding, interactions)
    foreground = X
    background = X[:500]

    # Full decomposition with grouped num and cat features
    Imap_inv = [list(range(d_num)), list(range(d_num, d_num+d_cat))]
    components = get_components_ebm(model, foreground, background, Imap_inv=Imap_inv, anchored=anchored)
    U = list(components.keys())[1:]
    components_2 = get_components_brute_force(h, foreground, background, Imap_inv=Imap_inv,
                                                            interactions=U, anchored=anchored)
    assert np.isclose(components[()], components_2[()]).all(), "Intercepts are not the same"
    for u in U:
        assert np.isclose(components[u], components_2[u]).all(), "Components are not the same"

    # Grid PDP/ICE for feature [0, 1]
    xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    foreground = np.column_stack((xx.ravel(), yy.ravel()))
    components = get_components_ebm(model, foreground, background, Imap_inv=[[0, 1]], anchored=anchored)
    components_2 = get_components_brute_force(h, foreground, background, Imap_inv=[[0, 1]], anchored=anchored)
    assert np.isclose(components[()], components_2[()]).all(), "Intercepts are not the same"
    assert np.isclose(components[(0,)], components_2[(0,)]).all(), "Components are not the same"


@pytest.mark.parametrize("anchored", [False, True])
def test_ebm_marketing(anchored):
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.model_selection import train_test_split
    from pyfd.data import get_data_marketing

    X, y, features = get_data_marketing(use_target_encoder=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, 
                                                stratify=y, random_state=42)
    model = ExplainableBoostingClassifier(random_state=0, interactions=5, max_bins=256)
    model.fit(X_train, y_train)

    # Reference data
    background = X_train[:200]
    # Compute the functional decomposition
    components = get_components_ebm(model, background, background, anchored=anchored)
    # Compute the functional decomposition
    U = list(components.keys())[1:]
    components_2 = get_components_brute_force(model.decision_function, background, background, 
                                              interactions=U, anchored=anchored)
    assert np.isclose(components[()], components_2[()]).all(), "Intercepts are not the same"
    for u in U:
        assert np.isclose(components[u], components_2[u]).all(), "Components are not the same"

    # Grouping day:month
    grouped_features = features.group([[5, 6]])
    # Compute the functional decomposition
    components = get_components_ebm(model, background, background, 
                                    Imap_inv=grouped_features.Imap_inv, 
                                    anchored=anchored)
    U = list(components.keys())[1:]
    # Compute the functional decomposition
    components_2 = get_components_brute_force(model.decision_function, background, background, 
                                              interactions=U, Imap_inv=grouped_features.Imap_inv, 
                                              anchored=anchored)
    assert np.isclose(components[()], components_2[()]).all(), "Intercepts are not the same"
    for u in U:
        assert np.isclose(components[u], components_2[u]).all(), "Components are not the same"



if __name__ == "__main__":
    # test_toy_ebm_grouping(3, 2, "regression", "identity", True, False)
    test_ebm_marketing()

