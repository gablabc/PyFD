""" Computing functional decompositions of black-box models """

import ctypes
import glob
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline

from .utils import check_Imap_inv, get_Imap_inv_from_pipeline, get_leaf_box
from .utils import ravel, powerset, setup_brute_force, setup_treeshap



#######################################################################################
#                                       Linear
#######################################################################################


SKLEARN_LINEAR = [LinearRegression, Ridge, LogisticRegression, PoissonRegressor]

def get_components_linear(h, foreground, background, Imap_inv=None):
    """
    Compute the Interventional Decomposition of Linear Model

    Parameters
    ----------
    h : model X -> R
        A sklearn LinearModel or a Pipeline with a LinearModel as the last layer
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition
    Imap_inv : List(List(int)), default=None
        A list of groups that represent a single feature. For instance `[[0, 1], [2]]` will treat
        the columns 0 and 1 as a single feature. The default approach is to treat each column as a feature
    
    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g. `decomposition[(0,)]`
        is a (Nf,) np.ndarray.
    """

    # Setup
    Imap_inv, D, _ = check_Imap_inv(Imap_inv, background.shape[1])
    if D > 1:
        assert foreground.shape[1] > 1, "When computing several h components, foreground must be a dataset"
    else:
        # If a line is provided we can compute the PDP
        if foreground.ndim == 1:
            foreground = foreground.reshape((-1, 1))

    # Check model type, identify the ML task, and get average prediction
    is_pipeline = False
    if type(h) == Pipeline:
        model_type = type(h.steps[-1][1])
        is_pipeline = True
    else:
        model_type = type(h)
    assert model_type in SKLEARN_LINEAR, "The predictor must be a linear/additive model"
    # For regression we explain the output
    if model_type in [LinearRegression, Ridge, PoissonRegressor]:
        h_emptyset_z = h.predict(background)
    # FOr classification we explain the logit
    else:
        h_emptyset_z = h.decision_function(background)
    decomposition = {}
    decomposition[()] = h_emptyset_z.mean()

    #If h is a Pipeline whose last layer is a linear model, we propagate the foreground and background
    # up to that point and we compute the Imap_inv up to the linear layer
    if is_pipeline:
        preprocessing = h[:-1]
        predictor = h[-1]
        background = preprocessing.transform(background)
        foreground = preprocessing.transform(foreground)
        Imap_inv = get_Imap_inv_from_pipeline(Imap_inv, preprocessing)
    else:
        predictor = h

    w = predictor.coef_.ravel()

    # Compute the additive components
    for j in range(D):
        Imap_inv_j = np.array(Imap_inv[j])
        # TODO this breaks if the processed data is sparse
        decomposition[(j,)] = np.sum((foreground[:, Imap_inv_j] - background[:, Imap_inv_j].mean(0)) * w[Imap_inv_j], axis=1)
    
    return decomposition


#######################################################################################
#                                    Model-Agnostic
#######################################################################################



def get_components_brute_force(h, foreground, background, Imap_inv=None, interactions=1, anchored=True, show_bar=True):
    """
    Compute the Anchored/Interventional Decomposition of any black box

    Parameters
    ----------
    h : model X -> R
        A callable black box `h(X)`.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the decomposition
    background : (Nb, d) np.ndarray
        The data points at which to anchor the decomposition
    Imap_inv : List(List(int)), default=None
        A list of groups that represent a single feature. For instance `[[0, 1], [2]]` will treat
        the columns 0 and 1 as a single feature. The default approach is to treat each column as a feature
    interactions : int, List(Tuple(int)), default=1
        The highest level of interactions to consider. If it is a list of tuples, then we compute all
        specified components in `interactions`.
    anchored : bool, default=True
        Flag to compute anchored decompositions or interventional decompositions. If anchored, a component
        is (Nf, Nb) and if interventional it is (Nf, 2) with final index indicating expectancy and std.
    
    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g. `decomposition[(1, 2, 3)]`
        returns the 3-way interactions between features 1, 2 and 3.
    """
    
    # Setup
    foreground, Imap_inv, iterator_, one_group = setup_brute_force(foreground, background, Imap_inv, interactions, show_bar)
    N_ref = background.shape[0]
    N_eval = foreground.shape[0]

    # Average prediction
    decomposition = {}

    # Compute all additive components  h_{i, z}(x) = h(x_i, z_-i) - h(z)
    # for all feature groups i in Imap_inv and points x,z
    h_emptyset_z = h(background)
    decomposition[()] = h_emptyset_z

    data_temp = np.copy(background)
    # Model-Agnostic Implementation (very costly in time and memory)
    # Compute h_{u, z}(x) = h(x_{u}, z_-{u}) - sum_{v\subset u} h_{u, z}(x)
    for key in iterator_():
        decomposition[key] = np.zeros((N_eval, N_ref))
        idx = ravel([Imap_inv[i] for i in key])
        for n in range(N_eval):
            if one_group:
                data_temp[:, idx] = foreground[n]
            else:
                data_temp[:, idx] = foreground[n, idx]
            h_ij_z = h(data_temp)
            decomposition[key][n] += h_ij_z
        # Reset the copied background
        data_temp[:,  idx] = background[:,  idx] 
        # Remove all contributions of subsets to get the interaction
        for subset in powerset(key):
            if subset not in decomposition:
                raise Exception("The provided interaction set is not closed downward")
            decomposition[key] -= decomposition[subset]
    if not anchored:
        for key, component in decomposition.items():
            if len(key) >= 1:
                decomposition[key] = np.column_stack((component.mean(1), component.std(1)))
    return decomposition



def get_components_adaptive(h, background, Imap_inv=None, tolerance=0.05, show_bar=True, precompute=None):
    """
    Compute the Anchored/Interventional Decomposition of any black box by iteratively exploring
    the lattice space of feature interactions.

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
        A precomputed decomposition containing all additive terms (i.e. `decomp[(i,)]`) can be provided to
        speed up the algorithm
    
    Returns
    -------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g. `decomposition[(1, 2, 3)]`
        returns the 3-way interactions between features 1, 2 and 3.
    """
    
    # Setup
    Imap_inv, D, is_fullpartition = check_Imap_inv(Imap_inv, background.shape[1])
    assert is_fullpartition, "In adaptive, Imap_inv must be a partition of the input columns"
    N = background.shape[0]
    data_temp = np.copy(background)

    # We compute the additive decomposition if it is not precomputed
    if precompute is None:
        # Average prediction
        decomposition = {}
        decomposition[()] = h(background)

        # Compute all additive components  h_{i, z}(x) = h(x_i, z_-i) - h(z)
        # for all feature groups i in Imap_inv and points x,z
        for i in tqdm(range(D), desc="Additive Components", disable=not show_bar):
            decomposition[(i,)] = np.zeros((N, N))
            for n in range(N):
                data_temp[:, Imap_inv[i]] = background[n, Imap_inv[i]]
                decomposition[(i,)][n] = h(data_temp) - decomposition[()]
            # Reset the copied background
            data_temp[:,  Imap_inv[i]] = background[:,  Imap_inv[i]]

    else:
        assert () in precompute.keys()
        assert precompute[()].shape == (N,)
        for i in range(D):
            assert (i,) in precompute.keys()
            assert precompute[(i,)].shape == (N, N)
        decomposition = deepcopy(precompute)
        
    
    # Setup for the adaptive interaction search
    variance = decomposition[()].var()
    h_proj = get_h_add(decomposition, anchored=True).mean(1)
    loss = np.mean( ( decomposition[()] - h_proj ) ** 2)
    # We get the interaction strenght for each of the current subsets u\in U
    psi, U = get_H_interaction(decomposition)
    # Data structure counting how many times we saw each super set
    seen_super_sets = {}
    # Store the decomposition without necessarily adding it to the dict
    temp_h_replace = np.zeros((N, N))

    # If specified, we compute all interactions up to a given size
    while loss / variance > tolerance:
        choice = np.argmax(psi)
        u = U[choice]
        
        # Never pick this point again
        psi[choice] = -1

        # Update the collection of supersets
        combine_candidates = [(i,) for i in range(D) if not i in u]
        for u_combine in combine_candidates:
            super_set = tuple(sorted(u_combine + u))
            # If never seen we add it to the dict
            if not super_set in seen_super_sets.keys():
                seen_super_sets[super_set] = 1
            # If already seen we increase its counter by one
            else:
                seen_super_sets[super_set] += 1
                # When all of its subsets are in U we can add it to U
                if seen_super_sets[super_set] == len(super_set):
                    
                    # Compute the component of the new added set
                    joint_idx = ravel([Imap_inv[i] for i in super_set])
                    for n in range(N):
                        data_temp[:, joint_idx] = background[n, joint_idx]
                        h_ij_z = h(data_temp)
                        temp_h_replace[n] = h_ij_z
                    
                    # Reset the copied background
                    data_temp[:,  joint_idx] = background[:,  joint_idx] 
                    
                    # Remove all contributions of subsets to get the interaction
                    for subset in powerset(super_set):
                        temp_h_replace -= decomposition[subset]
                    
                    # The component is essentially null so we do not add it
                    if np.mean(temp_h_replace.mean(1)**2) > 1e-8 * variance:
                        U.append(super_set)
                        decomposition[super_set] = np.copy(temp_h_replace)

                        # Update the loss
                        h_proj += decomposition[super_set].mean(1)
                        loss = np.mean( ( decomposition[()] - h_proj ) ** 2)
                        
                        if loss / variance <= tolerance:
                            break
                        
                        # Compute the psi to decide if we must explore the supersets
                        # of the newly added set
                        psi_ = (-1)**len(super_set) * decomposition[super_set].mean(0)
                        psi_ -= decomposition[super_set].mean(1)
                        psi = np.append(psi, np.mean(psi_**2))
    return decomposition



#######################################################################################
#                                    Tree Ensembles
#######################################################################################    



def get_component_tree(model, foreground, background, Imap_inv=None, anchored=False, algorithm='recurse'):
    """ 
    Compute the Anchored/Interventional Decomposition of a tree ensemble 
    (e.g. Random Forest and Gradient Boosted Trees)

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
    anchored : bool, default=False
        Flag to compute anchored decompositions or interventional decompositions.
    algorithm : string, default='recurse'
        The algorithm used to compute the decompositions, the options are
        - `recurse` with complexity `Nf Nb 2^min(depth, n_features)` can compute anchored and interventional
        - `leaf` with complexity `(Nf+Nb) 2^min(depth, n_features)` can only compute interventional

    Returns
    -------
    results : numpy.ndarray
        A (Nf, n_features) array if `anchored=False`, otherwise a (Nf, Nb, n_features)
        array of additive components.
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

    # Find the shared library, the path depends on the platform and Python version    
    project_root = os.path.dirname(__file__).split('pyfd')[0]
    libfile = glob.glob(os.path.join(project_root, 'treeshap*.so'))[0]

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
                decomp[(i,)] = np.stack((results[..., i].mean(-1), 
                                         results[..., i].std(-1)), axis=-1)
    
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
            decomp[(i,)] = np.stack((results[..., i], -1 * np.ones(Nx)), axis=-1)

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
            decomp[(i,)] = np.stack((results[..., i], -1 * np.ones(Nx)), axis=-1)
    else:
        raise Exception("Invalid algorithm, pick from `recurse` `leaf` or `waterfall`")
    
    return decomp



def get_PDP_PFI_importance(decomposition, groups=None, show_bar=True):
    """
    Compute PDP and PFI feature importance given an anchored decomposition

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored decomposition with foreground=background so that `decomposition[(0,)].shape = (N, N)`
    groups : (N,) np.ndarray, default=None
        An array of N integers (values in {0, 1, 2, n_groups-1}) representing the group index of each datum.
    show_bar : bool, default=True
        Show the progress bar

    Returns
    -------
    I_PDP : np.ndarray
        PDP feature importance. If `groups=None` then this array is (n_features,). Otherwise it has 
        shape (n_groups, n_features).
    I_PFI : np.ndarray
        PFI feature importance. If `groups=None` then this array is (n_features,). Otherwise it has 
        shape (n_groups, n_features).
    """

    # Get the additive decomposition
    keys = decomposition.keys()
    additive_keys = [key for key in keys if len(key)==1]
    D = len(additive_keys)
    shape_decomposition = decomposition[additive_keys[0]].shape
    assert shape_decomposition[0] == shape_decomposition[1], "The decomposition must be anchored with foreground=background"
    
    # Assert if explanations are regional
    if groups is None:
        n_groups = 1
    elif isinstance(groups, np.ndarray):
        n_groups = groups.max() + 1
        assert shape_decomposition[0] == len(groups), "Each foreground element must have a group index"
    else:
        raise Exception("Groups must be None or a numpy array")

    # No grouping
    if n_groups == 1:
        I_PDP = np.zeros(D)
        I_PFI = np.zeros(D)
        for d in tqdm(range(D), desc="PDP/PFI Importance", disable=not show_bar):
            I_PDP[d] = np.mean(decomposition[additive_keys[d]].mean(1)**2)
            I_PFI[d] = np.mean(decomposition[additive_keys[d]].mean(0)**2)
        return I_PDP, I_PFI, additive_keys
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
        return I_PDP, I_PFI, additive_keys



def get_H_interaction(decomposition):
    """
    Compute the H^2 statistics measuring interaction strenght between
    feature `i` and the remaining ones

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        An anchored decomposition with foreground=background so that `decomposition[(0,)].shape = (N, N)`

    Returns
    -------
    I_H : (n_features,) np.ndarray
        Array containing the H2 statistic for each feature
    """
    keys = decomposition.keys()
    additive_keys = [key for key in keys if len(key)==1]
    D = len(additive_keys)
    shape_decomposition = decomposition[additive_keys[0]].shape
    assert shape_decomposition[0] == shape_decomposition[1], "The decomposition must be anchored with foreground=background"
    I_H  = np.zeros(D)
    for d in range(D):
        I_H[d]  = np.mean((decomposition[additive_keys[d]].mean(0) + \
                            decomposition[additive_keys[d]].mean(1))**2)
    return I_H, additive_keys



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
        If anchored=True, this returns a (Nf, Nb) array. Otherwise, a (Nf,) array is returned
    """

    keys = decomposition.keys()
    additive_keys = [key for key in keys if len(key)==1]
    h_add = 0
    # Additive terms
    for key in additive_keys:
        if anchored:
            h_add += decomposition[key]
        else:
            h_add += decomposition[key][:, 0]
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
        is interventional
    foreground_preds : np.ndarray, default=None
        Array containing the model predictions at all foreground data points. When set
        to `None`, it is assumed that foreground=background and so these predictions can be 
        extracted from `decomposition[()]`.
    Returns
    -------
    coe : float
        The cost of exclusion which measures 'Lack of Additivity'
    """

    # We assume background-foreground in that case
    if foreground_preds is None:
        foreground_preds = decomposition[()]
    factor = 100 / np.std(foreground_preds)
    h_add = get_h_add(decomposition, anchored)
    if anchored:
        return factor * np.sqrt( np.mean( (foreground_preds - h_add.mean(1))**2 ) )
    else:
        return factor * np.sqrt( np.mean( (foreground_preds - h_add)**2 ) )

