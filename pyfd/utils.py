import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from itertools import chain, combinations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline

from shap.maskers import Independent
from shap.explainers import Tree


ADDITIVE_TRANSFORMS = [StandardScaler, MinMaxScaler, QuantileTransformer,
                       FunctionTransformer, KBinsDiscretizer, SplineTransformer, OneHotEncoder]



def safe_isinstance(obj, class_path_str):
    """
    Taken from https://github.com/shap/shap/blob/master/shap/utils/_general.py
    
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    -------
    bool: True if isinstance is true and the package exists, False otherwise

    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurrence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        #Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)))


def ravel(tuples):
    """
    turn a tuple of tuples ((a, b), (c,)) into a single tuple (a, b, c)
    turn a list of list [[a, b], [c]] into a single list [a, b, c]
    """
    out = []
    for seq_tuple in tuples:
        for element in seq_tuple:
            out.append(element)
    return out



def key_from_term(term, Imap_inv):
    """
    TODO 
    """
    D = len(Imap_inv)
    key = ()
    for idx in term:
        caught_feature = np.where([idx in Imap_inv[i] for i in range(D)])[0]
        if caught_feature.size == 1:
            caught_feature = int(caught_feature)
            if caught_feature not in key:
                key = tuple(sorted(key + (caught_feature,)))
    return key



def check_Imap_inv(Imap_inv, d):
    if Imap_inv is None:
        Imap_inv = [[i] for i in range(d)]
        is_full_partition = True
    else:
        assert type(Imap_inv) in (list, tuple), "Imap_inv must be a list or a tuple"
        assert type(Imap_inv[0]) in (list, tuple), "Elements of Imap_inv must be lists or tuples"
        raveled_Imap_inv = sorted(ravel(Imap_inv))
        assert min(raveled_Imap_inv) >= 0, "Imap_inv must have positive values"
        assert max(raveled_Imap_inv) < d, "Imap_inv cannot exceed the value of features"
        # Is the Imap_inv a full partition of the input space ?
        is_full_partition = raveled_Imap_inv == list(range(d))
    D = len(Imap_inv)
    return Imap_inv, D, is_full_partition



def get_Imap_inv_from_pipeline(Imap_inv, pipeline):
    Imap_inv_copy = deepcopy(Imap_inv)
    Imap_inv_return = deepcopy(Imap_inv)
    # Iterate over the whole pipeline
    for layer in pipeline:
        # Only certain layers are supported, other layers are assumed to be ``passthrough''
        if type(layer) in ADDITIVE_TRANSFORMS or type(layer) == ColumnTransformer:

            # Compute the list of transformers and the columns they act on
            if type(layer) in [SplineTransformer, OneHotEncoder]:
                transformers = [('layer', layer, range(layer.n_features_in_))]
            elif type(layer) == KBinsDiscretizer and layer.encode in ['onehot', 'onehot-dense']:
                transformers = [('layer', layer, range(layer.n_features_in_))]
            elif type(layer) == ColumnTransformer:
                transformers = layer.transformers_
            else:
                transformers = None

            # If transformers increase the number of columns, we must change Imap_inv_copy
            # This is done by composing Imap_inv_copy with Imap_inv_layer
            if transformers is not None:
                # We compute the Imap_inv relative to this layer
                Imap_inv_layer = [[] for _ in range(layer.n_features_in_)]

                curr_idx = 0
                for transformer in transformers:
                    # Splines maps a column to dim columns
                    if type(transformer[1]) == SplineTransformer:            
                        degree = transformer[1].degree
                        n_knots = transformer[1].n_knots
                        dim = n_knots + degree - 2 + int(transformer[1].include_bias)
                        # Iterate over all columns the transformer acts upon
                        for i in transformer[2]:
                            Imap_inv_layer[i] = list(range(curr_idx, curr_idx+dim))
                            curr_idx += dim
                    # OHE maps a column to n_categories columns
                    elif type(transformer[1]) == OneHotEncoder:
                        # Iterate over all columns the transformer acts upon
                        for idx, i in enumerate(transformer[2]):
                            n_categories = len(transformer[1].categories_[idx])
                            Imap_inv_layer[i] = list(range(curr_idx, curr_idx+n_categories))
                            curr_idx += n_categories
                    # BinsDiscretizer maps a column to n_bins columns
                    elif type(transformer[1]) == KBinsDiscretizer and \
                            transformer[1].encode in ['onehot', 'onehot-dense']:
                        # Iterate over all columns the transformer acts upon
                        for idx, i in enumerate(transformer[2]):
                            n_bins = transformer[1].n_bins_[idx]
                            Imap_inv_layer[i] = list(range(curr_idx, curr_idx+n_bins))
                            curr_idx += n_bins
                    # Other transformers map a column to a column
                    else:
                        for i in transformer[2]:
                            Imap_inv_layer[i] = [curr_idx]
                            curr_idx += 1

                # We map Imap_inv through the Imap_inv_layer
                for i, group in enumerate(Imap_inv_copy):
                    Imap_inv_return[i] = []
                    for j in group:
                        Imap_inv_return[i] += Imap_inv_layer[j]
                Imap_inv_copy = deepcopy(Imap_inv_return)
    
    return Imap_inv_return



def setup_linear(h, foreground, background, Imap_inv, acceptable_types):
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
    if safe_isinstance(h, ["sklearn.pipeline.Pipeline", "imblearn.pipeline.Pipeline"]):
        model_type = type(h.steps[-1][1])
        is_pipeline = True
    else:
        model_type = type(h)
    assert model_type in acceptable_types, "The predictor is not of the right class for this function"

    # If a PDP is computed, we must pad the foreground so
    # that it can be fed to the pipeline
    if not foreground.shape[1] == h.n_features_in_:
        foreground_pad = np.tile(background[0], (foreground.shape[0], 1))
        foreground_pad[:, Imap_inv[0]] = foreground
        foreground = foreground_pad
    
    # If h is a Pipeline whose last layer is a Linear/EBM layer, we propagate the foreground and background
    # up to that point and we compute the Imap_inv up to the Linear/EBM layer
    if is_pipeline:
        # IMB pipelines can be problematic when the last step does sampling, and so does not have
        # a transform method. A turnaround is to add a None step at the end.
        preprocessing = deepcopy(h[:-1])
        if safe_isinstance(h, "imblearn.pipeline.Pipeline"):
            preprocessing.steps.append(['predictor', None])
        predictor = h[-1]

        # Pass the foreground and background through the pipeline
        background = preprocessing.transform(background)
        foreground = preprocessing.transform(foreground)
        Imap_inv = get_Imap_inv_from_pipeline(Imap_inv, preprocessing)
    else:
        predictor = h

    return predictor, foreground, background, Imap_inv



def get_term_bin_weights(ebm, term_idx, bin_indexes, Nb):
    """ Adaptation of the make_bin_weights function of InterpretML """
    
    feature_idxs = ebm.term_features_[term_idx]
    multiple = 1
    dimensions = []
    for dimension_idx in range(len(feature_idxs) - 1, -1, -1):
        feature_idx = feature_idxs[dimension_idx]
        bin_levels = ebm.bins_[feature_idx]
        feature_bins = bin_levels[min(len(bin_levels), len(feature_idxs)) - 1]
        n_bins = len(feature_bins) + 3

        dimensions.append(n_bins)
        dim_data = deepcopy(bin_indexes[dimension_idx][:Nb])
        # dim_data = np.where(dim_data < 0, n_bins - 1, dim_data)
        if multiple == 1:
            flat_indexes = dim_data
        else:
            flat_indexes += dim_data * multiple
        multiple *= n_bins
    dimensions = tuple(reversed(dimensions))
    term_bin_weights = np.bincount(flat_indexes, minlength=multiple)
    term_bin_weights = term_bin_weights.astype(np.float64, copy=False)
    return term_bin_weights.reshape(dimensions) / Nb



def setup_brute_force(foreground, background, Imap_inv, interactions, show_bar):
    d = background.shape[1]
    if type(interactions) == int:
        assert interactions >= 1, "interactions must be 1, 2, 3, ..."
        assert interactions <= d, "interactions cannot be greater than the number of features"
        def iterator_():
            for cardinality in range(1, interactions+1):
                for key in tqdm(combinations(range(D), r=cardinality), desc="Functional Decomposition", disable=not show_bar):
                    yield key
    elif type(interactions) == list:
        assert type(interactions[0]) == tuple, "interactions should be a list of tuples"
        interactions = sorted(interactions, key=len)
        for i in range(len(interactions)):
            interactions[i] = tuple(interactions[i])
            def iterator_():
                for key in tqdm(interactions, desc="Functional Decomposition", disable=not show_bar):
                    yield key
    else:
        raise Exception("interactions must either be an integer or a list of tuples")
    
    # Setup Imap_inv
    Imap_inv, D, _ = check_Imap_inv(Imap_inv, d)

    # 1D foregrounds are reshaped as (N_eval, 1) arrays
    if foreground.ndim == 1:
        foreground = foreground.reshape((-1, 1))
    
    # Process foreground differently when we have one feature
    if D == 1:
        # The user is computing a PDP
        if foreground.shape[1] == len(Imap_inv[0]):
            # To keep the API consistent we unfortunately have to
            # pad out the foreground with zeros, even if a single
            # PDP is requested.
            foreground_pad = np.zeros((foreground.shape[0], d))
            foreground_pad[:, Imap_inv[0]] = foreground
            foreground = foreground_pad
        else:
            assert foreground.shape[1] == d, "When computing PDP, foreground must be a (Nf, d) or (Nf, group_size) dataset "
    else:
        assert foreground.shape[1] == d, "When computing several h components, foreground must be a (Nf, d) dataset"
    
    return foreground, Imap_inv, iterator_



def setup_treeshap(Imap_inv, foreground, background, model):

    # Map Imap_inv through the pipeline
    if safe_isinstance(model, ["sklearn.pipeline.Pipeline", "imblearn.pipeline.Pipeline"]):
        # IMB pipelines can be problematic when the last step does sampling, and so does not have
        # a transform method. A turnaround is to add a None step at the end.
        preprocessing = deepcopy(model[:-1])
        if safe_isinstance(model, "imblearn.pipeline.Pipeline"):
            preprocessing.steps.append(['predictor', None])
        model = model[-1]
        background = preprocessing.transform(background)
        foreground = preprocessing.transform(foreground)
        Imap_inv = get_Imap_inv_from_pipeline(Imap_inv, preprocessing)

    # Extract tree structure with the SHAP API
    masker = Independent(data=background, max_samples=background.shape[0])
    ensemble = Tree(model, data=masker).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']
    assert ensemble.node_sample_weight.flags['C_CONTIGUOUS']
    
    # All arrays must be C-Contiguous and DataFrames are not.
    if type(foreground) == pd.DataFrame or not foreground.flags['C_CONTIGUOUS']:
        foreground = np.ascontiguousarray(foreground)
    if type(background) == pd.DataFrame or not background.flags['C_CONTIGUOUS']:
        background = np.ascontiguousarray(background)
    
    Imap = np.zeros(background.shape[1], dtype=np.int32)
    for i, group in enumerate(Imap_inv):
        for column in group:
            Imap[column] = i
    return Imap, foreground, background, model, ensemble



def get_leaf_box(d, Nt, features, thresholds, left_child, right_child):
    global leaf_id

    M = np.max(np.sum(features<0, axis=-1))
    all_boxes = np.zeros((Nt, M, d, 2))
    for t in range(Nt):
        leaf_id = 0
        box = np.array([[-np.inf, np.inf] for _ in range(d)])
        leaf_box_recurse(box, 0, features[t], thresholds[t], left_child[t], right_child[t], all_boxes[t])

    boxes_min = np.ascontiguousarray(all_boxes[..., 0])
    boxes_max = np.ascontiguousarray(all_boxes[..., 1])
    return M, boxes_min, boxes_max


def leaf_box_recurse(box, node, features, thresholds, left_child, right_child, all_boxes):
    global leaf_id

    # Current state
    curr_feature = features[node]
    curr_threshold = thresholds[node]
    curr_left_child = left_child[node]
    curr_right_child = right_child[node]

    # Is a leaf
    if (curr_feature < 0):
        all_boxes[leaf_id] = box
        leaf_id += 1
    else:
        # Going left
        curr_box = deepcopy(box)
        if curr_threshold < box[curr_feature, 1]:
            curr_box[curr_feature, 1] = curr_threshold
        leaf_box_recurse(curr_box, curr_left_child, features, thresholds, left_child, right_child, all_boxes)
        
        # Going right
        curr_box = deepcopy(box)
        if box[curr_feature, 0] < curr_threshold:
            curr_box[curr_feature, 0] = curr_threshold
        leaf_box_recurse(curr_box, curr_right_child, features, thresholds, left_child, right_child, all_boxes)




def get_quantiles(x_i, bins):
    """Get quantiles from a feature in a dataset.

    Parameters
    ----------
    x_i : np.ndarray
        (N,) array containing the feature
    bins : int
        The number of quantiles is calculated as `bins + 1`.

    Returns
    -------
    quantiles : array-like
        Quantiles.
    bins : int
        Number of bins, `len(quantiles) - 1`. This may be lower than the original
        `bins` if identical quantiles were present.

    Raises
    ------
    ValueError
        If `bins` is not an integer.

    Notes
    -----
    When using this definition of quantiles in combination with a half open interval
    (lower quantile, upper quantile], care has to taken that the smallest observation
    is included in the first bin. This is handled transparently by `np.digitize`.

    """
    if not isinstance(bins, (int, np.integer)):
        raise ValueError(
            "Expected integer 'bins', but got type '{}'.".format(type(bins))
        )
    quantiles = np.unique(
        np.quantile(x_i, np.linspace(0, 1, bins + 1), interpolation="lower")
    )
    bins = len(quantiles) - 1
    return quantiles, bins