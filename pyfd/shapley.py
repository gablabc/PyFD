""" Module for computing Shapley Values """

import ctypes
import glob
import numpy as np
from tqdm import tqdm
import os
from .utils import check_Imap_inv, setup_treeshap, get_leaf_box, ravel, powerset



#######################################################################################
#                               Precomputed Decomposition
#######################################################################################



def shap_from_decomposition(decomposition):
    """
    Compute the Shapley Values from a provided functional decomposition.

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        The various components of the decomposition indexed via their feature subset e.g. 
        `decomposition[(1, 2, 3)]` returns the 3-way interactions between features 1, 2 and 3.
        Components can have shape (Nf,) or (Nf, Nb)
    
    Returns
    -------
    shapley_values : np.ndarray
        Shapley Values array of shape (Nf, Nb, n_features) if the decomposition is anchored and
        (Nf, n_features) otherwise. 
    """

    # Setup
    keys = decomposition.keys()
    D = len([key for key in keys if len(key)==1])
    shap_values = np.zeros(decomposition[(0,)].shape+(D,))
    for key in keys:
        if len(key) == 0:
            continue
        for feature in key:
            shap_values[..., feature] += decomposition[key] / len(key)
    return shap_values



#######################################################################################
#                                    Model-Agnostic
#######################################################################################


def permutation_shap(h, foreground, background, features, M=20, show_bar=True, reversed=True, return_nu_evals=False):
    """
    Approximate the Shapley Values of any black box by sampling M permutations

    Parameters
    ----------
    h : model X -> R
        A callable black box `h(X)`.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the shapley values.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the shapley values.
    features : Features
        A Features object that represents which columns of X are treated as groups.
    M : int, default=20
        Number of permutations to sample.
    show_bar : bool, default=True
        Flag to decide if progress bar is shown.
    reversed : bool, default=True
        Flag to sample a permutation and its reverse.
    return_nu_evals : bool, default=False
        Flag to return the number of calls `nu(S)` for the coallitional game.
    
    Returns
    -------
    shapley_values : (Nf, n_features) np.ndarray
        The interventional shapley values.
    """

    # Setup Imap_inv
    Imap_inv = features.Imap_inv
    Imap_inv, D, is_full_partition = check_Imap_inv(Imap_inv, background.shape[1])
    assert is_full_partition, "PermutationSHAP requires Imap_inv to be a partition of the input columns"
    N_eval = foreground.shape[0]
    if reversed:
        def permutation_generator(M, D):
            assert M%2==0, "Reversed Permutations require an even M"
            for _ in range(M//2):
                perm = np.random.permutation(D)
                yield perm
                yield perm[-1::-1]
    else:
        def permutation_generator(M, D):
            for _ in range(M):
                perm = np.random.permutation(D)
                yield perm
    
    # Compute the W matrix for SHAP value computations
    all_sets = dict({() : np.zeros(D)})
    # Go though all permutations to store all required sets
    for perm in permutation_generator(M, D):
        curr_set = []
        prev_tuple = None

        # Add the effect of the null coallition
        all_sets[()][perm[0]] -= 1
        
        # Going along the permutation
        for i in perm:
            curr_set.append(i)
            curr_set.sort()
            curr_tuple = tuple(curr_set)
            # Check if this curr_tuple has previously been used as key
            if curr_tuple not in all_sets:
                all_sets[curr_tuple] = np.zeros(D)
            all_sets[curr_tuple][i] += 1
            if prev_tuple is not None:
                all_sets[prev_tuple][i] -= 1
            # Keep track of the previous tuple
            prev_tuple = curr_tuple
    # All sets S on which we will evaluate nu(S)
    W = np.column_stack(list(all_sets.values())) / M
    interactions = list(all_sets.keys())
    S = len(interactions)

    # Setup arrays
    nu = np.zeros((S, N_eval))
    data_temp = np.copy(background)
    # Compute the weight matrix by visiting the lattice space
    for i, key in tqdm(enumerate(interactions), desc="Shapley Values", disable=not show_bar):
        cardinality = len(key)
        if cardinality == 0:
            nu[i] += h(background).mean()
        else:
            idx = ravel([Imap_inv[j] for j in key])
            for n in range(N_eval):
                data_temp[:, idx] = foreground[n, idx]
                nu[i][n] +=  h(data_temp).mean()
            # Reset the copied background
            data_temp[:,  idx] = background[:,  idx]

    if return_nu_evals:
        return W.dot(nu).T, S
    return W.dot(nu).T



def lattice_shap(h, foreground, background, features, interactions, show_bar=True, return_nu_evals=False):
    """
    Approximate the Shapley Values of any black box given a subsample of the lattice-space

    Parameters
    ----------
    h : model X -> R
        A callable black box `h(X)`.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the shapley values.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the shapley values.
    features : Features
        A Features object that represents which columns of X are treated as groups.
    interactions : List(Tuple(int))
        List of tuples representing the lattice space.
    show_bar : bool, default=True
        Flag to decide if progress bar is shown.
    return_nu_evals : bool, default=False
        Flag to return the number of calls `nu(S)` for the coallitional game.
    
    Returns
    -------
    shapley_values : (Nf, n_features) np.ndarray
        The interventional shapley values.
    """
    
    assert type(interactions) == list, "interactions must be a list of tuples"
    assert () in interactions
    S = len(interactions)

    # Setup Imap_inv
    Imap_inv = features.Imap_inv
    Imap_inv, D, is_full_partition = check_Imap_inv(Imap_inv, background.shape[1])
    assert is_full_partition, "LatticeSHAP requires Imap_inv to be a partition of the input columns"
    N_eval = foreground.shape[0]
    interactions_to_index = {interactions[i]:i for i in range(S)}

    # Setup arrays
    W = np.zeros((D, S))
    nu = np.zeros((S, N_eval))
    data_temp = np.copy(background)
    # Compute the weight matrix by visiting the lattice space
    for i, key in tqdm(enumerate(interactions), desc="Shapley Values", disable=not show_bar):
        cardinality = len(key)
        if cardinality == 0:
            nu[i] += h(background).mean()
        else:
            idx = ravel([Imap_inv[j] for j in key])
            for n in range(N_eval):
                data_temp[:, idx] = foreground[n, idx]
                nu[i][n] +=  h(data_temp).mean()
            # Reset the copied background
            data_temp[:,  idx] = background[:,  idx]
        if cardinality == 0:
            continue
        
        # Update the weight array
        W[key, i] += 1 / cardinality
        for subset in powerset(key):
            weight = (-1)**(cardinality - len(subset)) / cardinality
            W[key, interactions_to_index[subset]] += weight
    
    if return_nu_evals:
        return W.dot(nu).T, S
    return W.dot(nu).T



#######################################################################################
#                                    Tree Ensembles
#######################################################################################



def interventional_treeshap(model, foreground, background, features, anchored=False, algorithm="recurse"):
    """ 
    Compute the Interventional Shapley Values with the TreeSHAP algorithm.

    Parameters
    ----------
    model : model X -> R
        A tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.
    foreground : (Nf, d) np.ndarray
        The data points at which to evaluate the shapley values.
    background : (Nb, d) np.ndarray
        The data points at which to anchor the shapley values.
    features : Features
        A Features object that represents which columns of X are treated as groups.
    anchored : bool, default=False
        Flag to compute anchored shapley values or interventional shapley values.
    algorithm : string, default='recurse'
        The algorithm used to compute the shapley values, the options are
        - `recurse` with complexity `Nf Nb 2^min(depth, n_features)` can compute anchored and interventional
        - `leaf` with complexity `(Nf+Nb) 2^min(depth, n_features)` can only compute interventional.

    Returns
    -------
    results : np.ndarray
        A (Nf n_features) array of interventional SHAP values if `anchored=False`. 
        Otherwise a (Nf, Nb, n_features) array of anchored SHAP values.
    """

    sym = id(foreground) == id(background)

    if anchored and algorithm == 'leaf':
        raise Exception("Anchored decompositions are only supported by the `recurse` algorithm")
    
    # Setup
    Imap_inv = features.Imap_inv
    Imap_inv, _, is_full_partition = check_Imap_inv(Imap_inv, background.shape[1])
    assert is_full_partition, "TreeSHAP requires Imap_inv to be a partition of the input columns"
    Imap, foreground, background, model, ensemble = setup_treeshap(Imap_inv, foreground, background, model)
    
    # Shapes
    d = foreground.shape[1]
    Nx = foreground.shape[0]
    Nz = background.shape[0]
    Nt = ensemble.features.shape[0]
    n_features = np.max(Imap) + 1
    depth = ensemble.features.shape[1]

    # Values at each leaf
    values = np.ascontiguousarray(ensemble.values[..., -1])

    ####### Wrap C / Python #######

    # Find the shared library    
    libfile = glob.glob(os.path.join(os.path.dirname(__file__), 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    if algorithm == "recurse":
        # Where to store the output
        results = np.zeros((Nx, Nz, n_features)) if anchored else np.zeros((Nx, n_features)) 

        # Tell Python the argument and result types of function main_treeshap
        mylib.main_recurse_treeshap.restype = ctypes.c_int
        mylib.main_recurse_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                            ctypes.c_int, ctypes.c_int,
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            np.ctypeslib.ndpointer(dtype=np.int32),
                                            ctypes.c_int, ctypes.c_int,
                                            np.ctypeslib.ndpointer(dtype=np.float64)]

        # 3. call function mysum
        mylib.main_recurse_treeshap(Nx, Nz, Nt, foreground.shape[1], depth, foreground, background, 
                                Imap, ensemble.thresholds, values,
                                ensemble.features, ensemble.children_left,
                                ensemble.children_right, anchored, sym, results)
    elif algorithm == "leaf":
                # Where to store the output
        results = np.zeros((Nx, n_features))
        max_var = min(ensemble.max_depth, n_features)

        # Get the boundary boxes of all the leaves
        M, box_min, box_max = get_leaf_box(d, Nt, ensemble.features, ensemble.thresholds, 
                                        ensemble.children_left, ensemble.children_right)

        # Tell Python the argument and result types of function main_treeshap
        mylib.main_leaf_treeshap.restype = ctypes.c_int
        mylib.main_leaf_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
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
        mylib.main_leaf_treeshap(Nx, Nz, Nt, d, depth, M, max_var, foreground, background, Imap,
                                values, ensemble.features, ensemble.children_left, 
                                ensemble.children_right, box_min, box_max, 
                                results)
    
    else:
        raise Exception("Invalid algorithm, pick from `recurse` or `leaf`")

    return results



def taylor_treeshap(model, foreground, background, features):
    """ 
    Compute the Shapley-Taylor interaction indices by adapting the the TreeSHAP algorithm

    Parameters
    ----------
    model : model_object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.
    foreground : np.ndarray or pandas.DataFrame
        The foreground dataset is the set of all points whose prediction we wish to explain.
    background : np.ndarray or pandas.DataFrame
        The background dataset to use for integrating out missing features in the coallitional game.
    features : Features
        A Features object that represents which columns of X are treated as groups.

    Returns
    -------
    results : np.ndarray
        A (N_foreground, n_features, n_features) array
    """

    # Setup
    Imap_inv = features.Imap_inv
    Imap_inv, _, is_full_partition = check_Imap_inv(Imap_inv, background.shape[1])
    assert is_full_partition, "TreeSHAP requires Imap_inv to be a partition of the input columns"
    Imap, foreground, background, model, ensemble = setup_treeshap(Imap_inv, foreground, background, model)
    
    # Shapes
    d = foreground.shape[1]
    n_features = np.max(Imap) + 1
    depth = ensemble.features.shape[1]
    Nx = foreground.shape[0]
    Nz = background.shape[0]
    Nt = ensemble.features.shape[0]

    # Values at each leaf
    values = np.ascontiguousarray(ensemble.values[..., -1]/Nz)

    # Where to store the output
    results = np.zeros((Nx, n_features, n_features))

    ####### Wrap C / Python #######

    # Find the shared library
    libfile = glob.glob(os.path.join(os.path.dirname(__file__), 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_taylor_treeshap.restype = ctypes.c_int
    mylib.main_taylor_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_int,
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.int32),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.int32),
                                        np.ctypeslib.ndpointer(dtype=np.int32),
                                        np.ctypeslib.ndpointer(dtype=np.int32),
                                        np.ctypeslib.ndpointer(dtype=np.float64)]

    # 3. call function mysum
    mylib.main_taylor_treeshap(Nx, Nz, Nt, d, depth, foreground, background, 
                                Imap, ensemble.thresholds, values,
                                ensemble.features, ensemble.children_left,
                                ensemble.children_right, results)

    return results



#######################################################################################
#                                       Utilities
####################################################################################### 



def get_SHAP_importance(shapley_values, p=2, bootstrap_error=False):
    """
    Aggregate Shapley Values to get global feature importance.

    Parameters
    ----------
    shapley_values : np.ndarray
        Array of shape (Nf, Nb, n_features) or (Nf, n_features).
    bootstrap_error : bool, default=False
        Return boostrap -/+ 95% error estimates. This requires anchored SHAP values of
        shape (Nf, Nf, n_features) with foreground=background.

    Returns
    -------
    I_PDP : (n_features,) np.ndarray
        PDP feature importance.
    error_PFI : (2, n_features) np.ndarray, optional
        -/+ bootstrap errors for PFI importance
    """
    # Get the SHAP feature importance
    anchored = shapley_values.ndim == 3
    if anchored:
        I_SHAP = np.mean(np.abs(np.mean(shapley_values, axis=1))**p, axis=0)**(1/p)
    else:
        I_SHAP = np.mean(np.abs(shapley_values)**p, axis=0)**(1/p)
    if not bootstrap_error:
        return I_SHAP
    if not anchored:
        raise Exception("Boostrapping requires anchored Shapley Values")
    
    # Compute bootstrap error estimates
    N, M, D = shapley_values.shape
    Error_SHAP = np.zeros((2, D))
    for d in tqdm(range(D), desc="Bootstrap Error Estimate"):
        I_SHAP_distr = np.zeros((100,))
        for rep in range(10):
            samples = np.random.choice(range(N), size=(N, 1, 10))
            if N == M:
                samples_T = samples.reshape((1, N, 10))
            else:
                samples_T = np.random.choice(range(M), size=(1, M, 10))
            bootstrap_data = shapley_values[..., d][samples, samples_T]
            I_SHAP_distr[10*rep:10*(rep+1)] = np.mean(np.abs(bootstrap_data.mean(1))**p, axis=0)**(1/p)
        Error_SHAP[0, d] = max(0, I_SHAP[d] - np.quantile(I_SHAP_distr, q=0.05))
        Error_SHAP[1, d] = max(0, np.quantile(I_SHAP_distr, q=0.95) - I_SHAP[d])
    return I_SHAP, Error_SHAP
