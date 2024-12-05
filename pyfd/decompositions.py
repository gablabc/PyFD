""" Computing functional decompositions of black-box models """

import ctypes
import glob
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import SGDRegressor, SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC

from .utils import check_Imap_inv, get_leaf_box, ravel, powerset, key_from_term
from .utils import get_term_bin_weights, setup_linear, setup_brute_force, setup_treeshap



#######################################################################################
#                                 Linear & Additive
#######################################################################################



def get_components_linear(h, foreground, background, features):
    """
    Compute the Interventional Decomposition of a Linear Model.

    .. math:: h_{i, \mathcal{B}}(x_i) = \omega_i (x_i - \mathbb{E}_{z\sim\mathcal{B}}[z_i])

    Parameters
    ----------
    h : model X -> R
        A sklearn LinearModel or a Pipeline with a LinearModel as the last layer.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition.
    features : Features
        A Features object that represents which columns of X are treated as groups.

    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g. `decomposition[(0,)]`
        is a (Nf,) np.ndarray.
    """
    SKLEARN_LINEAR = [LinearRegression, Ridge, Lasso, ElasticNet,
                      SGDRegressor, SGDClassifier, LogisticRegression, LinearSVC]

    # Setup
    Imap_inv = features.Imap_inv
    predictor, foreground, background, Imap_inv = setup_linear(h, foreground, background, Imap_inv, SKLEARN_LINEAR)
    # For regression we explain the direct output
    if type(predictor) in [LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor]:
        h_emptyset_z = predictor.predict(background)
    # For classification we explain the logit
    else:
        h_emptyset_z = predictor.decision_function(background)
    decomposition = {}
    decomposition[()] = h_emptyset_z.mean()

    # Compute the additive components
    w = predictor.coef_.ravel()
    for j in range(len(Imap_inv)):
        Imap_inv_j = np.array(Imap_inv[j])
        # TODO this breaks if the processed data is sparse
        decomposition[(j,)] = np.sum((foreground[:, Imap_inv_j] - background[:, Imap_inv_j].mean(0)) * w[Imap_inv_j], axis=1)

    return decomposition



def get_components_ebm(h, foreground, background, features, anchored=True):
    """
    Compute the Interventional Decomposition of an Explainable Boosting Machine (EBM).

    .. math:: h_{i,\mathcal{B}}(x_i)  = h_i(x_i) - \mathbb{E}_{z\sim\mathcal{B}}[h_i(z_i)]

    Parameters
    ----------
    h : model X -> R
        A EBM model from interpret or a Pipeline with a EBM as the last layer.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition.
    features : Features
        A Features object that represents which columns of X are treated as groups.
    anchored : bool, default=True
        Flag to compute anchored decompositions or interventional decompositions. If anchored, a
        component is (Nf, Nb). If interventional, a component is (Nf,).

    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g.
        `decomposition[(0,)]` is a (Nf,) or (Nf, Nb) np.ndarray. This function returns all
        main effects and pair-wise interactions involving the provided features.
    """
    from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
    from interpret.glassbox._ebm._bin import eval_terms

    EBM = [ExplainableBoostingRegressor, ExplainableBoostingClassifier]

    # Setup
    Imap_inv = features.Imap_inv
    h, foreground, background, Imap_inv = setup_linear(h, foreground, background, Imap_inv, EBM)
    X = np.vstack((background, foreground))
    Nf = foreground.shape[0]
    Nb = background.shape[0]
    decomposition = {}
    decomposition[()] = h.intercept_ * np.ones((Nb,))
    component_shape = (Nf, Nb) if anchored else (Nf,)

    # Iterate over all terms in the EBM
    for term_idx, bin_indexes in eval_terms(X, Nf+Nb, h.feature_names_in_,
                            h.feature_types_in_, h.bins_, h.term_features_):

        # Find out to which subsets of features this term contributes to
        term = h.term_features_[term_idx]
        key = key_from_term(term, Imap_inv)
        if not key in decomposition.keys():
            decomposition[key] = np.zeros(component_shape)

        # () term h(z)
        term_scores = h.term_scores_[term_idx]
        scores = term_scores[tuple(bin_indexes)]
        hz = scores[:Nb]
        hx = scores[Nb:]
        # When anchored, we do broadcasting to get (Nf, Nb) components
        if anchored:
            hz = hz.reshape((1, -1))
            hx = hx.reshape((-1, 1))
        decomposition[()] += hz.ravel()
        del scores

        # Empty key only contribute to the () term
        if len(key) == 0:
            continue

        # For interventional, hz now represents E[h(z)]
        if not anchored:
            hz = hz.mean()

        # A main effect
        if len(term) == 1:
            # (i,) term is h(x) - h(z)
            decomposition[key] -= hz
            decomposition[key] += hx
        # A (i, j) pair-wise interaction
        else:
            if anchored:
                # (Nf, Nb) matrices
                hxizj = term_scores[bin_indexes[0][Nb:].reshape((-1, 1)),
                                    bin_indexes[1][:Nb].reshape((1, -1))]
                hzixj = term_scores[bin_indexes[0][:Nb].reshape((1, -1)),
                                    bin_indexes[1][Nb:].reshape((-1, 1))]
            else:
                # (Nf,) matrices
                term_bin_weights = get_term_bin_weights(h, term_idx, bin_indexes, Nb)
                hxizj = np.average(term_scores, axis=1, weights=term_bin_weights.sum(0))[bin_indexes[0][Nb:]]
                hzixj = np.average(term_scores, axis=0, weights=term_bin_weights.sum(1))[bin_indexes[1][Nb:]]

            if len(key) == 1:
                # A main effect is being computed but we must know if the
                # two idxs in the term are part of a single group
                # 1) Single Group
                if term[0] in Imap_inv[key[0]] and term[1] in Imap_inv[key[0]]:
                    decomposition[key] += hx - hz
                # 2) Main effect for first idx
                elif term[0] in Imap_inv[key[0]]:
                    decomposition[key] += hxizj - hz
                # 3) Main effect for second idx
                else:
                    decomposition[key] += hzixj - hz
            else:
                #  h(xi, zj) ___ h(xi, xj)
                #      |             |
                #  h(zi, zj) ___ h(zi, xj)
                #
                # (i, j) term h(x) - h(xi, zj) - h(zi, xj) + h(z)
                decomposition[key] += hx - hxizj - hzixj + hz
                # (i,) term  h(xi, zj) - h(z)
                i_key = (key[0],) if term[0] in Imap_inv[key[0]] else (key[1],)
                decomposition[i_key] -= hz
                decomposition[i_key] += hxizj
                # (j,) term  h(zi, xj) - h(z)
                j_key = (key[0],) if term[1] in Imap_inv[key[0]] else (key[1],)
                decomposition[j_key] -= hz
                decomposition[j_key] += hzixj

    return decomposition




#######################################################################################
#                                    Model-Agnostic
#######################################################################################


def _get_anchored_components_u(decomposition, h, key, Imap_inv, x_idxs, foreground, background):
    """
    Compute the Anchored Decomposition h_u(x) for a given u and at given x values

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g.
        `decomposition[(1, 2, 3)]` returns the 3-way interactions between features 1, 2 and 3.
    h : model X -> R
        A callable black box `h(X)`.
    key : Tuple(Int)
        The key that represents the set u at which to evaluate h_u.
    Imap_inv : List(List(int)), default=None
        A list of groups that represent a single feature. For instance `[[0, 1], [2]]` will treat
        the columns 0 and 1 as a single feature. The default approach is to treat each column as a feature.
    x_idxs : List(Int)
        The index of all foreground points at which to evaluate the decomposition. This parameter is useful
        when there are repetitions in feature values and so it is not necessary to loop over all foreground
        points.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition.

    Returns
    -------
    results : (len(x_idxs), Nb) np.ndarray
        The decomposition h_u anchored at all background points and evaluated at all foreground points.
    """
    N_eval = len(x_idxs)
    N_ref = background.shape[0]
    result = np.zeros((N_eval, N_ref))
    data_temp = np.copy(background)
    # Compute h_{u,k}(x) for all x to eval
    for i, x_idx in enumerate(x_idxs):
        # Annihilation property
        annihilated = np.zeros((N_ref,))
        for j in key:
            annihilated += (foreground[x_idx, Imap_inv[j]] == background[:, Imap_inv[j]]).all(axis=1)
        not_annihilated = np.where(annihilated == 0)[0]

        # Compute the component
        U = ravel([Imap_inv[j] for j in key])
        if not_annihilated.sum() > 0:
            data_temp[not_annihilated.reshape((-1, 1)), U] = foreground[x_idx, U]
            h_replace = h(data_temp[not_annihilated])
            result[i, not_annihilated] += h_replace
            # Remove all contributions of subsets to get the interaction
            for subset in powerset(key):
                if subset not in decomposition:
                    raise Exception("The provided interaction set is not closed downward")
                if len(subset) > 0:
                    result[i, not_annihilated] -= decomposition[subset][x_idx, not_annihilated]
                else:
                    result[i, not_annihilated] -= decomposition[subset][not_annihilated]
    return result



def get_components_brute_force(h, foreground, background, features, interactions=1, anchored=True, show_bar=False):
    """
    Compute the Anchored/Interventional Decomposition of any black box.

    .. math::

        h_{u,z}(x_u) = \sum_{v\subseteq u} (-1)^{|u|-|v|} h(x_v, z_{-v})

        h_{u,\mathcal{B}}(x_u) = \mathbb{E}_{z\sim \mathcal{B}} [h_{u,z}(x_u)]

    Parameters
    ----------
    h : model X -> R
        A callable black box `h(X)`.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition.
    features : Features
        A Features object that represents which columns of X are treated as groups.
    interactions : int, List(Tuple(int)), default=1
        The highest level of interactions to consider. If it is a list of tuples, then we compute all
        specified components in `interactions`.
    anchored : bool, default=True
        Flag to compute anchored decompositions or interventional decompositions. If anchored, a
        component is (Nf, Nb). If interventional, a component is (Nf,).
    show_bar : bool, default=False
        Flag to decide if progress bar is shown.

    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g.
        `decomposition[(1, 2, 3)]` returns the 3-way interactions between features 1, 2 and 3.
    """

    # Setup
    Imap_inv = features.Imap_inv
    foreground, Imap_inv, iterator_ = setup_brute_force(foreground, background, Imap_inv, interactions, show_bar)
    N_eval = foreground.shape[0]
    N_ref = background.shape[0]
    decomposition = {}

    # Compute the intercept
    decomposition[()] = h(background)

    # It is possible that foreground points have the same feature value. For low order interactions
    # it could be better to iterate over all unique feature values rather than iterating over
    # all x in foregroud
    unique_feature_values = []
    for i in Imap_inv:
        if len(i) == 1:
            unique, indices = np.unique(foreground[:, i], return_index=True)
            # Is it worst to iterate on unique feature value?
            if len(unique) < N_eval:
                unique_feature_values.append([unique, indices])
            else:
                unique_feature_values.append(None)
        else:
            unique_feature_values.append(None)

    # Compute the components h_u for all u in U
    for key in iterator_():
        # Iterate over unique values of x_i
        if len(key) == 1 and unique_feature_values[key[0]] is not None:
            unique, indices = unique_feature_values[key[0]]
            x_idxs = indices
            result = _get_anchored_components_u(decomposition, h, key, Imap_inv, x_idxs, foreground, background)
            decomposition[key] = np.zeros((N_eval, N_ref))
            for i, value in enumerate(unique):
                select = foreground[:, Imap_inv[key[0]][0]]==value
                decomposition[key][select] = result[i]
        # Or Iterate over all N_eval foreground points
        else:
            x_idxs = np.arange(N_eval)
            decomposition[key] = _get_anchored_components_u(decomposition, h, key, Imap_inv, x_idxs,
                                                                            foreground, background)
    if anchored:
        return decomposition
    else:
        return get_interventional_from_anchored(decomposition)



def get_components_adaptive(h, background, features, tolerance=0.05, show_bar=False, precompute=None):
    """
    Compute the AnchoredDecomposition of any black box 

    .. math::

        h_{u,z}(x_u) = \sum_{v\subseteq u} (-1)^{|u|-|v|} h(x_v, z_{-v})

    by iteratively exploring the lattice space of feature interactions. This function assumes 
    that foreground=background in order to exploit the duality between averaging a component row-wise and column-wise.

    Parameters
    ----------
    h : model X -> R
        A callable black box `h(X)`.
    background : (Nb, d) np.ndarray
        The data points at which to anchor and evaluate the decomposition
    features : Features
        A Features object that represents which columns of X are treated as groups.
    tolerance : float, default=0.05
        Stop exploring the lattice space when the explained variance exceeds `1-tolerance` of the total variance.
    show_bar : bool, default=False
        Flag to decide if progress bar is shown.
    precompute: dict{Tuple: np.ndarray}, default=None
        A precomputed decomposition containing all additive terms (i.e. `decomp[(i,)]`) can be provided
        to speed up the algorithm. The components must be (Nb, Nb).

    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g.
        `decomposition[(1, 2, 3)]` returns the (Nb, Nb) the 3-way interactions between features
        1, 2 and 3.
    """

    # Setup
    Imap_inv = features.Imap_inv
    Imap_inv, D, is_fullpartition = check_Imap_inv(Imap_inv, background.shape[1])
    assert is_fullpartition, "In adaptive, Imap_inv must be a partition of the input columns"
    N = background.shape[0]

    # Compute the additive decomposition if it is not precomputed
    if precompute is None:
        decomposition = get_components_brute_force(h, background, background, features, show_bar=show_bar)
    else:
        assert () in precompute.keys()
        assert precompute[()].shape == (N,)
        for i in range(D):
            assert (i,) in precompute.keys()
            assert precompute[(i,)].shape == (N, N)
        decomposition = deepcopy(precompute)

    # Setup for lattice space search
    variance = decomposition[()].var()
    h_proj = get_h_add(decomposition, anchored=True).mean(1)
    loss = np.mean( ( decomposition[()] - h_proj ) ** 2)
    # We get the interaction strenght for each of the current subsets u\in U
    Psi, U = get_H_interaction(decomposition, return_keys=True)
    # Data structure counting how many times a candidate was proposed by its direct children
    candidates = {}

    # Iterate until the reconstruction loss is low
    while loss / variance > tolerance:
        choice = np.argmax(Psi)
        u_star = U[choice]

        # Never pick this node again
        Psi[choice] = -1

        # Update the collection of candidates
        combine_candidates = [(k,) for k in range(D) if not k in u_star]
        for u_combine in combine_candidates:
            u_candidate = tuple(sorted(u_combine + u_star))
            # Candidate never seen so add it to the dict
            if not u_candidate in candidates.keys():
                candidates[u_candidate] = 1
            else:
                # Candidate already seen so increase its counter by one
                candidates[u_candidate] += 1
                # Candidate has been proposed by all its direct children?
                if candidates[u_candidate] == len(u_candidate):

                    # Compute the component H^u of the new candidate set
                    H_u_candidate = _get_anchored_components_u(decomposition, h, u_candidate, Imap_inv,
                                                                np.arange(N), background, background)

                    # The component should not be null
                    if np.mean(H_u_candidate.mean(1)**2) > 1e-8 * variance:
                        U.append(u_candidate)
                        decomposition[u_candidate] = H_u_candidate

                        # Compute the Psi to decide if the node should be extended
                        Psi_new = (-1)**len(u_candidate) * H_u_candidate.mean(0)
                        Psi_new -= H_u_candidate.mean(1)
                        Psi = np.append(Psi, np.mean(Psi_new**2))

                        # Update the loss
                        h_proj += decomposition[u_candidate].mean(1)
                        loss = np.mean( ( decomposition[()] - h_proj ) ** 2)

    return decomposition



#######################################################################################
#                                    Tree Ensembles
#######################################################################################    



def get_components_tree(model, foreground, background, features, anchored=False, algorithm='recurse'):
    """
    Compute the additve terms of the Anchored/Interventional Decomposition of a tree ensemble 
    (e.g. Random Forest and Gradient Boosted Trees).

    .. math::

        h_{i,z}(x_i) = h(x_i, z_{-i}) - h(z)

        h_{i,\mathcal{B}}(x_i) = \mathbb{E}_{z\sim \mathcal{B}} [h_{i,z}(x_i)]

    Parameters
    ----------
    model : model X -> R
        A tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition
    features : Features
        A Features object that represents which columns of X are treated as groups.
    anchored : bool, default=True
        Flag to compute anchored decompositions or interventional decompositions. If anchored, a
        component is (Nf, Nb). If interventional, a component is (Nf,).
    algorithm : string, default='recurse'
        The algorithm used to compute the decompositions, the options are

        - `recurse` with complexity `Nf Nb 2^min(depth, n_features)` can compute anchored and interventional
        - `leaf` with complexity `(Nf+Nb) 2^min(depth, n_features)` can only compute interventional

    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g.
        `decomposition[(0,)]` returns the main effect of feature 0.
    """

    sym = id(foreground) == id(background)
    if anchored and algorithm in ['leaf', 'waterfall']:
        raise Exception("Anchored decompositions are only supported by the `recurse` algorithm")

    # Setup
    Imap_inv = features.Imap_inv
    Imap_inv, D, is_full_partition = check_Imap_inv(Imap_inv, background.shape[1])
    # We complete the partition with a `fake` feature if necessary
    if not is_full_partition:
        Imap_inv = deepcopy(Imap_inv)
        Imap_inv.append([])
        Imap_inv_ravel = ravel(Imap_inv)
        for k in range(foreground.shape[1]):
            if not k in Imap_inv_ravel:
                Imap_inv[-1].append(k)
    Imap, foreground, background, model, ensemble = setup_treeshap(Imap_inv, foreground, background, model)

    # Shapes
    d = foreground.shape[1]
    n_features = np.max(Imap) + 1
    depth = ensemble.features.shape[1]
    Nx = foreground.shape[0]
    Nz = background.shape[0]
    Nt = ensemble.features.shape[0]

    # Values at each leaf
    values = np.ascontiguousarray(ensemble.values[..., -1])

    # 0-th component
    decomp = {}
    preds = ensemble.predict(background) if ensemble.num_outputs == 1 else ensemble.predict(background)[..., -1]
    decomp[()] = preds

    ####### Wrap C / Python #######

    # Find the shared library
    libfile = glob.glob(os.path.join(os.path.dirname(__file__), 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    if algorithm == "recurse":
        # Where to store the output
        results = np.zeros((Nx, Nz, n_features))

        # Tell Python the argument and result types of function main_treeshap
        mylib.main_recurse_additive.restype = ctypes.c_int
        mylib.main_recurse_additive.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int,
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            ctypes.c_int,
                                            np.ctypeslib.ndpointer(dtype=np.float64)]

        # 3. call function mysum
        mylib.main_recurse_additive(Nx, Nz, Nt, d, depth, foreground, background,
                                Imap, ensemble.thresholds, values,
                                ensemble.features, ensemble.children_left,
                                ensemble.children_right, sym, results)

        for i in range(D):
            if anchored:
                decomp[(i,)] = results[..., i]
            else:
                decomp[(i,)] = results[..., i].mean(1)

    elif algorithm == "leaf":
        # Where to store the output
        results = np.zeros((Nx, n_features))

        # Get the boundary boxes of all the leaves
        M, box_min, box_max = get_leaf_box(d, Nt, ensemble.features, ensemble.thresholds,
                                        ensemble.children_left, ensemble.children_right)

        # Tell Python the argument and result types of function main_treeshap
        mylib.main_leaf_additive.restype = ctypes.c_int
        mylib.main_leaf_additive.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64)]

        # 3. call function mysum
        mylib.main_leaf_additive(Nx, Nz, Nt, d, depth, M, foreground, background, Imap,
                                values, ensemble.features, ensemble.children_left,
                                ensemble.children_right, box_min, box_max,
                                results)

        # We cannot compute standard deviation with this estimator
        for i in range(D):
            decomp[(i,)] = results[..., i]

    elif algorithm == "waterfall":
        # Where to store the output
        results = np.zeros((Nx, n_features))

        max_depth = ensemble.max_depth
        # Tell Python the argument and result types of function main_treeshap
        mylib.main_add_waterfallshap.restype = ctypes.c_int
        mylib.main_add_waterfallshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.int32),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.int32),
                                                np.ctypeslib.ndpointer(dtype=np.int32),
                                                np.ctypeslib.ndpointer(dtype=np.int32),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64)]

        # 3. call function mysum
        mylib.main_add_waterfallshap(Nx, Nz, Nt, d, depth, max_depth, foreground, background, Imap,
                                ensemble.thresholds, values, ensemble.features, ensemble.children_left,
                                ensemble.children_right, ensemble.node_sample_weight, results)

        # We cannot compute standard deviation with this estimator
        for i in range(D):
            decomp[(i,)] = results[..., i]
    else:
        raise Exception("Invalid algorithm, pick from `recurse` `leaf` or `waterfall`")

    return decomp



#######################################################################################
#                                       Utilities
####################################################################################### 



def get_PDP_PFI_importance(decomposition, variance=False, bootstrap_error=False, return_keys=False):
    """
    Compute PDP and PFI feature importance given an anchored decomposition.
    
    .. math::

        \Phi^{PDP}_i(h) = \mathbb{E}_{x\sim \mathcal{F}} [ h_{i,\mathcal{B}}(x_i)^2]
        
        \Phi^{PFI}_i(h) = \mathbb{E}_{x\sim \mathcal{F}} [ (\sum_{u:i\in u} h_{u,\mathcal{B}}(x_u))^2]

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored decomposition so that `decomposition[(0,)].shape = (Nf, Nb)`.
    variance : bool, default=False
        return the variance-based importance metrics PDP-Variance and Marginal-Sobol.
        Both of which are inviariant to correlation between feature j and the remaining features.
    bootstrap_error : bool, default=False
        Return boostrap -/+ 95% error estimates.
    return_keys : bool, default=False
        whether to return the additive keys associated with each interaction index.

    Returns
    -------
    I_PDP : (n_features,) np.ndarray
        PDP feature importance.
    I_PFI : (n_features,) np.ndarray
        PFI feature importance.
    error_PDP : (2, n_features) np.ndarray, optional
        -/+ bootstrap errors for PDP importance,
    error_PFI : (2, n_features) np.ndarray, optional
        -/+ bootstrap errors for PFI importance,
    additive_keys : List(List(int)), optional
        The key associated with each feature importance.
    """

    # Get the additive decomposition
    keys = decomposition.keys()
    additive_keys = [key for key in keys if len(key)==1]
    additive_keys = sorted(additive_keys, key=lambda x: x[0])
    D = len(additive_keys)
    shape_decomposition = decomposition[additive_keys[0]].shape
    assert len(shape_decomposition) == 2, "The decomposition must be anchored"

    # Feature importance
    I_PDP = np.zeros(D)
    I_PFI = np.zeros(D)
    for d in range(D):
        if variance:
            I_PDP[d] = np.sqrt(decomposition[additive_keys[d]].mean(1).var())
            I_PFI[d] = np.sqrt(decomposition[additive_keys[d]].var(0).mean())
        else:
            I_PDP[d] = np.sqrt(np.mean(decomposition[additive_keys[d]].mean(1)**2))
            I_PFI[d] = np.sqrt(np.mean(decomposition[additive_keys[d]].mean(0)**2))

    if not bootstrap_error:
        if return_keys:
            return I_PDP, I_PFI, additive_keys
        else:
            return I_PDP, I_PFI

    # Bootstrap error estimates
    N, M = shape_decomposition
    Error_PDP = np.zeros((2, D))
    Error_PFI = np.zeros((2, D))
    for d in tqdm(range(D), desc="Bootstrap Error Estimates"):
        I_PDP_distr = np.zeros((100,))
        I_PFI_distr = np.zeros((100,))
        for rep in range(10):
            samples = np.random.choice(range(N), size=(N, 1, 10))
            if N == M:
                samples_T = samples.reshape((1, N, 10))
            else:
                samples_T = np.random.choice(range(M), size=(1, M, 10))
            boostrapped_data = decomposition[additive_keys[d]][samples, samples_T]
            if variance:
                I_PDP_distr[10*rep:10*(rep+1)] = np.sqrt(boostrapped_data.mean(1).var(0))
                I_PFI_distr[10*rep:10*(rep+1)] = np.sqrt(boostrapped_data.var(0).mean(0))
            else:
                I_PDP_distr[10*rep:10*(rep+1)] = np.sqrt(np.mean(boostrapped_data.mean(1)**2, axis=0))
                I_PFI_distr[10*rep:10*(rep+1)] = np.sqrt(np.mean(boostrapped_data.mean(0)**2, axis=0))
        Error_PDP[0, d] = max(0, I_PDP[d] - np.quantile(I_PDP_distr, q=0.05))
        Error_PDP[1, d] = max(0, np.quantile(I_PDP_distr, q=0.95) - I_PDP[d])
        Error_PFI[0, d] = max(0, I_PFI[d] - np.quantile(I_PFI_distr, q=0.05))
        Error_PFI[1, d] = max(0, np.quantile(I_PFI_distr, q=0.95) - I_PFI[d])
    if return_keys:
        return I_PDP, I_PFI, Error_PDP, Error_PFI, additive_keys
    return I_PDP, I_PFI, Error_PDP, Error_PFI



def get_H_interaction(decomposition, return_keys=False):
    """
    Compute the H^2 statistics measuring interaction strenght between feature `i` and the remaining ones.

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored decomposition with foreground=background so that `decomposition[(0,)].shape = (N, N)`.
    return_keys : bool, default=False
        Flag to return the additive keys associated with each interaction index.

    Returns
    -------
    I_H : (n_features,) np.ndarray
        Array containing the H2 statistic for each feature
    additive_keys : List(List(int))
        The key associated with each interaction index.
    """
    keys = decomposition.keys()
    additive_keys = [key for key in keys if len(key)==1]
    D = len(additive_keys)
    shape_decomposition = decomposition[additive_keys[0]].shape
    assert len(shape_decomposition) == 2, "The decomposition must be anchored"
    assert shape_decomposition[0] == shape_decomposition[1], "The decomposition must have foreground=background"

    I_H  = np.zeros(D)
    for d in range(D):
        I_H[d]  = np.mean((decomposition[additive_keys[d]].mean(0) + \
                            decomposition[additive_keys[d]].mean(1))**2)
    if return_keys:
        return I_H, additive_keys
    return I_H



def get_h_add(decomposition, anchored=True, all_subsets=False):
    """
    Compute additive decomposition `h_add(x) = sum_i h_i(x)` evaluated at each foreground point x.

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored/interventional decomposition
    anchored : bool, default=True
        If True then the decomposition is anchored and if False the decomposition
        is interventional.
    all_subsets : bool, default=False
        If True, then returns the summation over all subsets sum_u h_u(x). This is
        useful when we estimate the lattice space and want to know the resulting error.

    Returns
    -------
    h_add : np.ndarray
        The additive decomposition summing the intercept and all main effects.
        If anchored=True, it is a (Nf, Nb) array. Otherwise, a (Nf,) array is returned.
    """

    keys = decomposition.keys()
    if all_subsets:
        additive_keys = [key for key in keys if len(key)>=1]
    else:
        additive_keys = [key for key in keys if len(key)==1]
    h_add = 0
    # Additive terms
    for key in additive_keys:
        h_add += decomposition[key]
    # Reference term
    if anchored:
        h_add += decomposition[()].reshape((1, -1))
    else:
        h_add += decomposition[()].mean()
    return h_add



def get_CoE(decomposition, foreground_preds=None, all_subsets=False):
    """
    Compute the Cost of Exclusion

    .. math:: \mathbb{E}_{x\sim\mathcal{F}}[(h(x) - \sum_{u:|u|\leq 1} h_{u,\mathcal{B}}(x_u)))^2 ]

    which is nothing more than the average squared error between the model and its additive decomposition.

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        The components of the decomposition indexed via their feature subset e.g.
        `decomposition[(0,)]` is an array of shape (Nf, Nb) or (Nf,). Can be a
        List(dict) of length `n_regions`.
    foreground_preds : np.ndarray, default=None
        Array containing the model predictions at all foreground data points. When set
        to `None`, it is assumed that foreground=background and so these predictions can be
        extracted from `decomposition[()]`. If `decomposition` is a List, then this must also
        be a list with the same length.
    all_subsets : bool, default=False
        If True, then returns the summation over all subsets sum_u h_u(x). This is
        useful when we estimate the lattice space and want to know the resulting error.

    Returns
    -------
    coe : float
        The Cost of Exclusion which measures interaction strength.
    """

    # Check the decomposition and foreground_preds
    if type(decomposition) == dict:
        decomposition = [decomposition]
        assert foreground_preds is None or type(foreground_preds) == np.ndarray
        if type(foreground_preds) == np.ndarray:
            foreground_preds = [foreground_preds]
    elif type(decomposition) == list:
        assert type(decomposition[0]) == dict, "decomposition must be a list of dict"
        assert foreground_preds is None or type(foreground_preds) == list
    else:
        raise Exception("decomposition must be a dict or a list of dict")
    anchored = decomposition[0][(0,)].ndim == 2
    n_regions = len(decomposition)

    # Set the foreground_preds to the intercept
    if foreground_preds is None:
        foreground_preds = [decomposition[r][()] for r in range(n_regions)]

    # At this point, decomposition and foreground_preds are both lists of length n_regions
    weights = np.zeros((n_regions,))
    CoEs = np.zeros((n_regions,))

    # Iterate over all regions
    for r in range(n_regions):
        # Get the additive decomposition intercept + sum_i h_i
        h_add = get_h_add(decomposition[r], anchored, all_subsets)
        weights[r] = len(foreground_preds[r])
        if anchored:
            CoEs[r] =  np.mean( (foreground_preds[r] - h_add.mean(1))**2 )
        else:
            CoEs[r] =  np.mean( (foreground_preds[r] - h_add)**2 )

    # Do a weighted average across regions
    weights = weights / np.sum(weights)
    # The CoE is normalized by the total model variance (accross all regions)
    factor = 100 / np.var( np.concatenate(foreground_preds, axis=None) )
    return factor * np.average(CoEs, weights=weights)



def get_interventional_from_anchored(decomposition):
    """
    Transform an anchored decomposition into an interventional one to save memory.

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored decomposition `decomposition[(0,)].shape=(Nf, Nb)`.

    Returns
    -------
    decomposition_ : dict{Tuple: np.ndarray}
        An interventional decomposition `decomposition[(0,)].shape=(Nf,)`.
    """

    decomposition_ = {}
    for key in decomposition.keys():
        if len(key)== 0:
            decomposition_[key] = deepcopy(decomposition[key])
        else:
            assert decomposition[key].ndim == 2, "The decomposition must be anchored"
            decomposition_[key] = decomposition[key].mean(1)
    return decomposition_



def get_regional_decompositions(decomposition, foreground_region_idx, background_region_idx, n_regions):
    """
    Transform an anchored decomposition into a list of regional anchored decompositions.

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored decomposition `decomposition[(0,)].shape=(Nf, Nb)`.
    foreground_region_idx : np.ndarray
        A `(Nf,)` array storing the region index of each foreground point.
    background_region_idx : np.ndarray
        A `(Nb,)` array storing the region index of each background point.
    n_regions : int
        The number of regions

    Returns
    -------
    regional_decompositions : List( dict{Tuple: np.ndarray} )
        A `(n_regions,)` list containing anchored decompositions restricted to the specific regions.
    """
    assert len(decomposition[(0,)].shape) == 2, "The decomposition must be anchored"
    regional_decompositions = []
    for region in range(n_regions):
        regional_decompositions.append({})
        for key in decomposition.keys():
            if len(key) == 0:
                regional_decompositions[-1][()] = decomposition[key][np.where(background_region_idx==region)[0]]
            else:
                foreground_select = np.where(foreground_region_idx==region)[0].reshape((-1, 1))
                background_select = np.where(background_region_idx==region)[0].reshape((1, -1))
                regional_decompositions[-1][key] = decomposition[key][foreground_select, background_select]
    return regional_decompositions
