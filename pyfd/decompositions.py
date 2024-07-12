""" Computing functional decompositions of black-box models """

import ctypes
import glob
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, PoissonRegressor

from .utils import check_Imap_inv, get_leaf_box, ravel, powerset, key_from_term
from .utils import get_term_bin_weights, setup_linear, setup_brute_force, setup_treeshap



#######################################################################################
#                                 Linear & Additive
#######################################################################################



def get_components_linear(h, foreground, background, Imap_inv=None):
    """
    Compute the Interventional Decomposition of Linear Model

    Parameters
    ----------
    h : model X -> R
        A sklearn LinearModel or a Pipeline with a LinearModel as the last layer.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition.
    Imap_inv : List(List(int)), default=None
        A list of groups that represent a single feature. For instance `[[0, 1], [2]]` will treat
        the columns 0 and 1 as a single feature. The default approach is to treat each column as a feature.
    
    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g. `decomposition[(0,)]`
        is a (Nf,) np.ndarray.
    """

    SKLEARN_LINEAR = [LinearRegression, Ridge, LogisticRegression, PoissonRegressor]

    # Setup
    predictor, foreground, background, Imap_inv = setup_linear(h, foreground, background, Imap_inv, SKLEARN_LINEAR)
    # For regression we explain the direct output
    if type(predictor) in [LinearRegression, Ridge, PoissonRegressor]:
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



def get_components_ebm(h, foreground, background, Imap_inv=None, anchored=True):
    """
    Compute the Interventional Decomposition of an Explainable Boosting Machine (EBM)

    Parameters
    ----------
    h : model X -> R
        A EBM model from interpret or a Pipeline with a EBM as the last layer.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition.
    Imap_inv : List(List(int)), default=None
        A list of groups that represent a single feature. For instance `[[0, 1], [2]]` will treat
        the columns 0 and 1 as a single feature. The default approach is to treat each column as a feature.
    anchored : bool, default=True
        Flag to compute anchored decompositions or interventional decompositions. If anchored, a 
        component is (Nf, Nb). If interventional, a component is (Nf,).
    
    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g. 
        `decomposition[(0,)]` is a (Nf,) of (Nf, Nb) np.ndarray. This function returns all
        main effects and pair-wise interactions involving the Imap_inv groups.
    """
    from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
    from interpret.glassbox._ebm._bin import eval_terms
    
    EBM = [ExplainableBoostingRegressor, ExplainableBoostingClassifier]

    # Setup
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
                # -h(zi, xj) ___ h(xi, xj)
                #      |             |
                #  h(zi, zj) ___ h(xi, zj)
                #
                # (i, j) term h(x) - h(xi, zj) - h(zi, xj) + h(z)
                decomposition[key] += hx - hxizj - hzixj + hz
                # (i,) term  h(xi, zj) - h(z)
                decomposition[key[:1]] -= hz
                decomposition[key[:1]] += hxizj
                # (j,) term  h(zi, xj) - h(z)
                decomposition[key[1:]] -= hz
                decomposition[key[1:]] += hzixj
    
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



def get_components_brute_force(h, foreground, background, Imap_inv=None, interactions=1, anchored=True, show_bar=False):
    """
    Compute the Anchored/Interventional Decomposition of any black box

    Parameters
    ----------
    h : model X -> R
        A callable black box `h(X)`.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition.
    Imap_inv : List(List(int)), default=None
        A list of groups that represent a single feature. For instance `[[0, 1], [2]]` will treat
        the columns 0 and 1 as a single feature. The default approach is to treat each column as a feature.
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



def get_components_adaptive(h, background, Imap_inv=None, tolerance=0.05, show_bar=False, precompute=None):
    """
    Compute the Anchored/Interventional Decomposition of any black box by iteratively exploring
    the lattice space of feature interactions. This function assumes that foreground=background
    in ordre to exploit the duality between averaging a component row-wise and column-wise.

    Parameters
    ----------
    h : model X -> R
        A callable black box `h(X)`.
    background : (Nb, d) np.ndarray
        The data points at which to anchor and evaluate the decomposition
    Imap_inv : List(List(int)), default=None
        A list of groups that represent a single feature. For instance `[[0, 1], [2]]` will treat
        the columns 0 and 1 as a single feature. The default approach is to treat each column as a feature
    interactions : int, List(Tuple(int)), default=1
        The highest level of interactions to consider. If it is a list of tuples, then we compute all
        specified components in `interactions`.
    tolerance : float, default=0.05
        Stop exploring the lattice space when the explained variance exceeds `1-tolerance` of the total variance.
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
    Imap_inv, D, is_fullpartition = check_Imap_inv(Imap_inv, background.shape[1])
    assert is_fullpartition, "In adaptive, Imap_inv must be a partition of the input columns"
    N = background.shape[0]

    # Compute the additive decomposition if it is not precomputed
    if precompute is None:
        decomposition = get_components_brute_force(h, background, background, Imap_inv, show_bar=show_bar)
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



def get_components_tree(model, foreground, background, Imap_inv=None, anchored=False, algorithm='recurse'):
    """ 
    Compute the Anchored/Interventional Decomposition of a tree ensemble 
    (e.g. Random Forest and Gradient Boosted Trees).

    Parameters
    ----------
    model : model X -> R
        A tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition
    Imap_inv : List(List(int)), default=None
        A list of groups that represent a single feature. For instance `[[0, 1], [2]]` will treat
        the columns 0 and 1 as a single feature. The default approach is to treat each column as a feature.
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



def get_PDP_PFI_importance(decomposition, groups=None, return_keys=False, variance=False, show_bar=False):
    """
    Compute PDP and PFI feature importance given an anchored decomposition

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored decomposition so that `decomposition[(0,)].shape = (Nf, Nb)`.
    groups : (N,) np.ndarray, default=None
        An array of N integers (values in {0, 1, 2, n_groups-1}) representing the group index of 
        each datum. Is only applicable if foreground=background and the decomposition has shape (N, N).
    return_keys : bool, default=False
        whether to return the additive keys associated with each interaction index.
    variance : bool, default=False
        return the variance-based importance metrics PDP-Variance and Marginal-Sobol. Both of which are inviariant
        to the correlation between feature j and the remaining features.
    show_bar : bool, default=True
        Show the progress bar.

    Returns
    -------
    I_PDP : np.ndarray
        PDP feature importance. If `groups=None` then this array is (n_features,). Otherwise it has 
        shape (n_groups, n_features).
    I_PFI : np.ndarray
        PFI feature importance. If `groups=None` then this array is (n_features,). Otherwise it has 
        shape (n_groups, n_features).
    additive_keys : List(List(int))
        The key associated with each feature importance.
    """

    # Get the additive decomposition
    keys = decomposition.keys()
    additive_keys = [key for key in keys if len(key)==1]
    additive_keys = sorted(additive_keys, key=lambda x: x[0])
    D = len(additive_keys)
    shape_decomposition = decomposition[additive_keys[0]].shape
    assert len(shape_decomposition) == 2, "The decomposition must be anchored"
    
    # Assert if explanations are regional
    if groups is None:
        n_groups = 1
    elif isinstance(groups, np.ndarray):
        assert shape_decomposition[0] == shape_decomposition[1], "Must have foreground=background when passing groups"
        n_groups = groups.max() + 1
        assert shape_decomposition[0] == len(groups), "Each instance must have a group index"
    else:
        raise Exception("Groups must be None or a numpy array")

    # No grouping
    if n_groups == 1:
        I_PDP = np.zeros(D)
        I_PFI = np.zeros(D)
        for d in tqdm(range(D), desc="PDP/PFI Importance", disable=not show_bar):
            if variance:
                I_PDP[d] = decomposition[additive_keys[d]].mean(1).var()
                I_PFI[d] = decomposition[additive_keys[d]].var(0).mean()
            else:
                I_PDP[d] = np.mean(decomposition[additive_keys[d]].mean(1)**2)
                I_PFI[d] = np.mean(decomposition[additive_keys[d]].mean(0)**2)
    # Separate feature importance for each group
    else:
        I_PDP = np.zeros((n_groups, D))
        I_PFI = np.zeros((n_groups, D))
        for group_id in range(n_groups):
            select = np.where(groups==group_id)[0].reshape((-1, 1))
            for d in tqdm(range(D), desc="PDP/PFI Importance", disable=not show_bar):
                H = decomposition[additive_keys[d]][select, select.T]
                I_PDP[group_id, d] = np.mean(H.mean(1)**2)
                I_PFI[group_id, d] = np.mean(H.mean(0)**2)
                if variance:
                    I_PDP[group_id, d] = H.mean(1).var()
                    I_PFI[group_id, d] = H.var(0).mean()
                else:
                    I_PDP[group_id, d] = np.mean(H.mean(1)**2)
                    I_PFI[group_id, d] = np.mean(H.mean(0)**2)
    
    if return_keys:
        return I_PDP, I_PFI, additive_keys
    return I_PDP, I_PFI


def get_H_interaction(decomposition, return_keys=False):
    """
    Compute the H^2 statistics measuring interaction strenght between
    feature `i` and the remaining ones

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored decomposition with foreground=background so that `decomposition[(0,)].shape = (N, N)`
    return_keys : bool, default=False
        whether to return the additive keys associated with each interaction index
    
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



def get_h_add(decomposition, anchored=True):
    """
    Compute additive decomposition `h_add(x) = sum_i h_i(x)`
    evaluated at each foreground point x

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored/interventional decomposition
    anchored : bool, default=True
        If True then the decomposition is anchored and if False the decomposition
        is interventional

    Returns
    -------
    h_add : np.ndarray
        The additive decomposition summing the intercept and all main effects.
        If anchored=True, is is a (Nf, Nb) array. Otherwise, a (Nf,) array is returned
    """

    keys = decomposition.keys()
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



def get_CoE(decomposition, anchored=True, foreground_preds=None):
    """
    Compute Cost of Exclusion `CoE = Ex( (h(x) - h_add(x))^2 )`

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored/interventional decomposition
    anchored : bool, default=True
        If True then the decomposition is anchored and if False the decomposition
        is interventional.
    foreground_preds : np.ndarray, default=None
        Array containing the model predictions at all foreground data points. When set
        to `None`, it is assumed that foreground=background and so these predictions can be 
        extracted from `decomposition[()]`.
    Returns
    -------
    coe : float
        The cost of exclusion which measures 'Lack of Additivity'
    """

    # We assume background=foreground in that case
    if foreground_preds is None:
        foreground_preds = decomposition[()]
    factor = 100 / np.var(foreground_preds)
    h_add = get_h_add(decomposition, anchored)
    if anchored:
        return factor * np.mean( (foreground_preds - h_add.mean(1))**2 )
    else:
        return factor * np.mean( (foreground_preds - h_add)**2 )


def get_interventional_from_anchored(decomposition):
    """
    Transform an anchored decomposition into an interventional one to
    save memory.
    
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
