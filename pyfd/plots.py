import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import combinations
from graphviz import Digraph
from math import ceil
from warnings import warn

from .decompositions import get_interventional_from_anchored
from .fd_trees import GADGET_PDP, PDP_PFI_Tree


def setup_pyplot_font(size=11, use_latex=False):
    from matplotlib import rc
    rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':size})
    if use_latex:
        rc('text', usetex=True)
        from matplotlib import rcParams
        rcParams["text.latex.preamble"] = r"\usepackage{bm}\usepackage{amsfonts}"



color_dict = {}
color_dict["default"] = {'zero' : [255, 255, 255], 
                         'pos':  [0, 102, 255], 
                         'neg' : [255, 69, 48]}
color_dict["DEEL"] = {'zero' : [255, 255, 255], 
                      'pos':  [0, 69, 138],
                      'neg' : [255, 69, 48]}
COLORS = ['blue', 'red', 'green', 'orange',
          'violet', 'brown', 'cyan', 'olive']



def abs_map_xerr(phi, xerr):
    """
    Map xerr on phi to xerr on |phi|

    Arguments
    ---------
    phi : (stacked, n_features) np.ndarray
    xerr : (stacked, 2, n_features) np.ndarray

    Returns
    -------
    abs_xerr (stacked, 2, n_features) nd.ndarray
    """
    stacked = phi.shape[0]
    cross_origin = ~( np.sign(phi-xerr[:, 0, :]) == np.sign(phi+xerr[:, 1, :]) ) # (stacked, n_features)
    # Min and Max of CIs
    map_bottom = np.abs(phi - xerr[:, 0, :])  # (stacked, n_features)
    map_top = np.abs(phi + xerr[:, 1, :])     # (stacked, n_features)
    min_CI  = np.minimum(map_bottom, map_top) # (stacked, n_features)
    max_CI  = np.maximum(map_bottom, map_top) # (stacked, n_features)
    # Minimum of CIs that cross origin is zero
    min_CI[cross_origin] = 0

    # Transform a CI to xerr
    return np.abs(np.abs(phi).reshape(stacked, 1, -1) - np.stack((min_CI, max_CI), axis=1))



def check_bar_args(phis, feature_labels, xerr):
    """ 
    After this function, phis is a (stacked, n_features) numpy array while
    xerr is a (stacked, 2, n_features) array
    """
    # Are there multiple feature labels?
    assert type(feature_labels) == list, "feature_labels must be a list"
    if type(feature_labels[0]) == list:
        assert type(feature_labels[0][0] ) == str, "feature_labels must be a `List(List(str))`"
        n_features = len(feature_labels[0])
        multiple_labels = True
    else:
        assert type(feature_labels[0]) == str, "feature_labels must be `List(str)`"
        n_features = len(feature_labels)
        multiple_labels = False
    
    # Check phis
    if type(phis) in [list, tuple]:
        assert type(phis[0]) == np.ndarray, "phis must be a list of np.ndarray"
        phis = np.vstack(phis)
    else:
        assert type(phis) == np.ndarray, "phis must be a np.ndarray"
        if phis.ndim == 1:
            phis = phis.reshape((1, -1))
    assert phis.shape[1] == n_features, "phis must involved `n_features` features"
    stacked_bars = phis.shape[0]
    
    # Check xerr
    if xerr is not None:
        if type(xerr) in [list, tuple]:
            assert len(xerr) == stacked_bars, "xerr must be a list of lenght `stacked`"
            assert type(xerr[0]) == np.ndarray, "xerr must contain np.ndarrays"
            assert xerr[0].shape in [(n_features,), (2, n_features)]
            if xerr[0].shape == (n_features, ):
                xerr = [np.vstack((error, error)) for error in xerr]
            xerr = np.stack(xerr, axis=0)
        else:
            assert type(xerr) == np.ndarray
            assert xerr.shape in [(n_features,), (2, n_features)]
            assert stacked_bars == 1, "xerr can only be passed as np.ndarray if no bars are stacked"
            if xerr.shape == (n_features, ):
                xerr = np.vstack((xerr, xerr)).reshape((1, 2, -1))
            else:
                xerr = xerr.reshape((1, 2, -1))
    return phis, xerr, multiple_labels



def bar(phis, feature_labels, threshold=None, xerr=None, absolute=False, ax=None, color=None):
    """
    Plot Feature Importance/Attribution bars

    Parameters
    ----------
    phis : (n_features,) or (stacks, n_features) `np.ndarray`
        Feature importance to plot as bars. If a `List(np.ndarray)` is provided, or
        the array is 2 dimensional, then bars are placed side-by-side.
    feature_labels : `List(string)`
        Name of the features shown on the y-axis. If a `[List(string), List(string)]`
        is provided, then the labels are shown on the left and right.
    threshold: `float`, default=None
        Show a threshold of significance
    xerr: `np.ndarray` or `List(np.ndarray)`, default=None
        If not `None`, add horizontal errorbars to the bars.
        When a list is provided, the error bars are shown for each stacked bar.
        The values are +/- sizes relative to the data:
        - shape(N,): symmetric +/- values for each bar
        - shape(2, N): Separate - and + values for each bar. First row
            is the lower errors, the second row is the upper errors.
    
    absolute `bool`, default=False
        Rank with respect to the absolute value of the importance
    ax : `pyplot.axis`
        Add the plots to a pre-existing axis
    """
    # Check the input arguments
    phis, xerr, multiple_labels = check_bar_args(phis, feature_labels, xerr)
    stacked_bars, n_features = phis.shape

    # Plotting the abs value of the phis
    if absolute:
        bar_mapper = lambda x : np.abs(x)
        if xerr is not None:
            xerr = abs_map_xerr(phis, xerr)
    else:
        bar_mapper = lambda x : x
        
    ordered_features = np.argsort(bar_mapper(phis[0]))
    y_pos = np.arange(len(ordered_features))
    
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Draw a line for the origin when necessary
    negative_phis = (phis < 0).any() and not absolute
    if negative_phis:
        ax.axvline(0, 0, 1, color="k", linestyle="-", linewidth=1, zorder=1)

    # Show the treshold of significant amplitude
    if threshold:
        ax.axvline(threshold, 0, 1, color="k", linestyle="--", linewidth=2, zorder=1)
        if negative_phis:
            ax.axvline(-threshold, 0, 1, color="k", linestyle="--", linewidth=2, zorder=1)

    # draw the bars
    bar_width = 0.7 / stacked_bars
    shift = [0] * stacked_bars
    if stacked_bars % 2 == 0:
        shift =  [ (i+0.5)*bar_width for i in range(stacked_bars//2)[::-1]]\
                +[-(i+0.5)*bar_width for i in range(stacked_bars//2)]
    else:
        shift =  [ (i+1)*bar_width for i in range(stacked_bars//2)[::-1]] + [0]\
                +[-(i+1)*bar_width for i in range(stacked_bars//2)]

    # Get DEEL colors
    if color is None:
        colors = deepcopy(color_dict["DEEL"])
        colors['pos'] = np.array(colors['pos'])/255.
        colors['neg'] = np.array(colors['neg'])/255.
    else:
        colors = {}
        colors['pos'] = color
        colors['neg'] = color

    # Plot the bars with err
    for s in range(stacked_bars):
        # Error bars
        if xerr is not None:
            error_bar = xerr[s][:, ordered_features]
        else:
            error_bar = None
        ax.barh(
            y_pos + shift[s], bar_mapper(phis[s, ordered_features]),
            bar_width, xerr=error_bar, align='center',
            color=[colors['neg'] if phis[s, ordered_features[j]] <= 0
                    else colors['pos'] for j in range(len(y_pos))],
            edgecolor=(0.88,0.89,0.92), capsize=5, alpha=1-0.75/stacked_bars*s)

    # Set the y-ticks and labels
    if multiple_labels:
        yticklabels = [feature_labels[0][j] for j in ordered_features]
        ax.set_yticks(ticks=list(y_pos))
        ax.set_yticklabels(yticklabels, fontsize=15)

        yticklabels = [feature_labels[1][j] for j in ordered_features]
        ax2 = ax.twinx()
        ax2.set_yticks(ticks=list(y_pos))
        ax2.set_yticklabels(yticklabels, fontsize=15)
        ax2.set_ybound(*ax.get_ybound())
    else:
        yticklabels = [feature_labels[j] for j in ordered_features]
        ax.set_yticks(ticks=list(y_pos))
        ax.set_yticklabels(yticklabels, fontsize=15)

    # put horizontal lines for each feature row
    for i in range(n_features):
        ax.axhline(i, color="k", lw=0.5, alpha=0.5, zorder=-1)

    xmin,xmax = ax.get_xlim()
    ax.set_ylim(-0.5, n_features-0.5)

    if negative_phis:
        ax.set_xlim(xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05)
    else:
        ax.set_xlim(xmin, xmax + (xmax-xmin)*0.05)

    plt.gcf().tight_layout()



def get_curr_axis(n_rows, n_cols, ax, iter):
    if n_rows == 1:
        if n_cols == 1:
            curr_ax = ax
        else:
            curr_ax = ax[iter]
    else:
        if n_cols == 1:
            curr_ax = ax[iter]
        else:
            curr_ax = ax[iter//n_cols][iter%n_cols]
    return curr_ax


def partial_dependence_plot(decomposition, foreground, background, features, idxs=None,
                            groups_method=None, rules=None, fd_trees_kwargs={}, centered=True,
                            figsize=None, n_cols=5, plot_hist=False, normalize_y=True, alpha=0.01):
    
    # If no idxs is provided, we plot all features
    if idxs is None:
        idxs = range(len(features))
        Imap_inv = deepcopy(features.Imap_inv)
    else:
        Imap_inv = deepcopy([features.Imap_inv[i] for i in idxs])
    for i in range(len(idxs)):
        assert len(Imap_inv[i]) == 1, "No feature grouping in PDP plots"
        Imap_inv[i] = Imap_inv[i][0]
    
    anchored = decomposition[(0,)].shape == (foreground.shape[0], background.shape[0])
    additive_keys = [(idx,) for idx in idxs]
    d = len(additive_keys)

    if anchored:
        y_min = min([np.percentile(decomposition[key], 1) for key in additive_keys])
        y_max = max([np.percentile(decomposition[key], 99) for key in additive_keys])
        importance = np.array([np.mean(decomposition[key].mean(0)**2) for key in additive_keys])
    else:
        y_min = min([decomposition[key].min() for key in additive_keys])
        y_max = max([decomposition[key].max() for key in additive_keys])
        importance = np.array([np.var(decomposition[key]) for key in additive_keys])
    idxs_ordered = np.argsort(-importance)

    delta_y = (y_max-y_min)
    y_min = y_min - delta_y*0.01
    y_max = y_max + delta_y*0.01

    if len(idxs) < n_cols:
        n_cols = len(idxs)
        n_rows = 1
    else:
        n_rows = ceil(d / n_cols)
    _, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for iter, i in enumerate(idxs_ordered):
        # Get current axis to plot
        curr_ax = get_curr_axis(n_rows, n_cols, ax, iter)

        # Key represents the name of the component
        key = additive_keys[i]
        # To which column of X it corresponds
        column = Imap_inv[i]
        x_min = foreground[:, column].min()
        x_max = foreground[:, column].max()
        # What feature is being studied
        feature = features.feature_objs[idxs[i]]

        if plot_hist:
            # TODO plot better histograms when categorical or integer
            _, _, rects = curr_ax.hist(foreground[:, column], bins=20, rwidth=0.95, color="gray", alpha=0.6, bottom=y_min)
            max_height = max([h.get_height() for h in rects])
            target_max_height = 0.5 * delta_y
            for r in rects:
                r.set_height(target_max_height*r.get_height()/max_height)
        
        if not anchored:
            sorted_idx = np.argsort(foreground[:, column])
            curr_ax.plot(foreground[sorted_idx, column], decomposition[key][sorted_idx], 'k-')
        else:
            plot_legend = False
            if groups_method is None:
                colors = ['k']
                rules = "all"
                n_groups = 1
                groups_foreground = np.zeros(foreground.shape[0])
                groups_background = np.zeros(background.shape[0])
            elif callable(groups_method):
                colors = COLORS
                assert rules is not None, "When providing groups you must also provide their rule description"
                n_groups = len(rules)
                groups_foreground = groups_method(foreground)
                groups_background = groups_method(background)
            elif groups_method in ["gadget-pdp", "pdp-pfi"]:
                colors = COLORS
                assert id(foreground) ==  id(background) ,"FDTrees requires background=foreground"
                # Fit a FDTree to reduce interactions involving the feature `column`
                decomp_copy = {}
                decomp_copy[()] = decomposition[()]
                decomp_copy[(0,)] = decomposition[key]
                background_slice = np.delete(background, column, axis=1)
                features_slice = features.remove([column])
                fd_tree = GADGET_PDP(features=features_slice, **fd_trees_kwargs) if groups_method == "gadget-pdp" \
                    else  PDP_PFI_Tree(features=features_slice, **fd_trees_kwargs) 
                fd_tree.fit(background_slice, decomp_copy)
                rules = fd_tree.rules(use_latex=True)
                groups_background = fd_tree.predict(background_slice)
                groups_foreground = groups_background
                n_groups = fd_tree.n_groups
                plot_legend = n_groups > 1
            else:
                raise Exception("Invalid grouping parameter")

            # For each group plot the anchored components in color
            n_curves_max = 500 // n_groups
            for group_id in range(n_groups):
                group_foreground = foreground[groups_foreground==group_id]
                sorted_idx = np.argsort(group_foreground[:, column])
                select_f = np.where(groups_foreground==group_id)[0][sorted_idx].reshape((-1, 1))
                select_b = np.where(groups_background==group_id)[0].reshape((-1, 1))
                H = decomposition[key][select_f, select_b.T]
                # Center the curves w.r.t the xaxis
                if centered:
                    H = H - H.mean(0)
                # Otherwise show the pdp/ice plus intercept
                # else:
                #     H += decomposition[()][select_b.ravel()]
                x = group_foreground[sorted_idx, column]
                # Plot at most 200 ICE curves
                curr_ax.plot(x, H[:, :n_curves_max], colors[group_id], alpha=alpha)
                curr_ax.plot(x, H.mean(1), 'k', linewidth=3)
                curr_ax.plot(x, H.mean(1), colors[group_id], label=rules[group_id], linewidth=2)
        
            if plot_legend:
                curr_ax.legend(fontsize=12, framealpha=1)
        
        # xticks labels for categorical data
        if feature.type in ["bool", "ordinal", "nominal"]:
            if feature.type in ["ordinal", "nominal"]:
                categories = feature.cats
                # Truncate names if too long
                # if len(categories) > 5:
                # categories = [name[:3] for name in categories]
                rotation = 90
            else:
                categories = [False, True]
                rotation = 0
            curr_ax.set_xticks(np.arange(len(categories)), labels=categories, rotation=rotation)
        
        # Set ticks and labels
        curr_ax.grid('on')
        curr_ax.set_xlabel(feature.name)
        if iter%n_cols == 0:
            curr_ax.set_ylabel("Local Feature Attribution")
        curr_ax.set_xlim(x_min, x_max)
        if normalize_y:
            curr_ax.set_ylim(y_min, y_max)
            if not iter%n_cols == 0:
                curr_ax.yaxis.set_ticklabels([])
    # Remove unused axes
    iter += 1
    while iter < n_rows * n_cols:
        get_curr_axis(n_rows, n_cols, ax, iter).set_axis_off()
        iter += 1


def attrib_scatter_plot(decomposition, shapley_values, foreground, features, idxs=None,
                        normalize_y=True, figsize=None, n_cols=5):
    """ 
    Plot the PDP/SHAP local feature attributions as a function of x_i.

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        The components of the decomposition indexed via their feature subset e.g. 
        `decomposition[(0,)]` is an array of shape (Nf, Nb) or (Nf,). Can be a
        List(dict) of length `n_regions`.
    shapley_values : np.ndarray
        The anchored (Nf, Nb, d) or interventional (Nf, d) shapley values. Can be a
        List(np.ndarray) of length `n_regions`.
    foreground : (Nf, d) np.ndarray
        The foreground data. Can be a list List(np.ndarray) of length `n_regions`.
    features : Features
        The features object.
    idxs : List(int) or int, default=None
        The index of the features to plot. When 
        - `None`, all non-group features are plotted.
        - `List(int)`, the non-group features with indices in List are plotted.
        - `int`, the top-idxs non-group features are plotted.
    
    normalize_y : bool, default=True
        Have the same y axis for all plots.
    figsize : List(int), default=None
        The size (Nx, Ny) of the figure.
    n_cols : int, default=5
        The number of columns in the subplot.
    """
    if idxs is None or type(idxs) == int:
        top_k = idxs if type(idxs) == int else len(features)
        idxs = list(range(len(features)))
        Imap_inv = deepcopy(features.Imap_inv)
    else:
        top_k = len(idxs)
        Imap_inv = deepcopy([features.Imap_inv[i] for i in idxs])
    # Remove grouped features
    for i in range(len(idxs)-1, -1, -1):
        if len(Imap_inv[i]) > 1:
            warn("Grouped Features Ignored")
            del idxs[i]
            del Imap_inv[i]
        else:
            Imap_inv[i] = Imap_inv[i][0]
    
    # Lists are passed for regional plots
    if type(decomposition) == list:
        assert type(shapley_values) == list and type(foreground) == list, "Provide lists of shap values and foregrounds"
        assert len(shapley_values) == len(decomposition), "Must have a consistent number of regions"
        assert len(foreground) == len(decomposition), "Must have a consistent number of regions"
        colors = COLORS
    # Otherwise non-regional explanations are shown
    else:
        assert type(decomposition) == dict
        assert type(shapley_values) == np.ndarray
        assert type(foreground) == np.ndarray
        decomposition = [decomposition]
        shapley_values = [shapley_values]
        foreground = [foreground]
        colors = ["grey"]
    n_regions = len(decomposition)

    # We need Interventional Decompositions and Shapley values
    if shapley_values[0].ndim == 3:
        shapley_values_copy = [0] * n_regions
        for r in range(n_regions):
            shapley_values_copy[r] = shapley_values[r].mean(1)
        shapley_values = shapley_values_copy
    if decomposition[0][(0,)].ndim == 2:
        decomposition_copy = [0] * n_regions
        for r in range(n_regions):
            decomposition_copy[r] = get_interventional_from_anchored(decomposition[r])
        decomposition = decomposition_copy
    
    # Min/max and feature importance
    y_min = min([phi[..., idxs].min() for phi in shapley_values])
    y_max = max([phi[..., idxs].max() for phi in shapley_values])
    importance = np.max(np.stack([np.mean(phi[..., idxs]**2, axis=0) for phi in shapley_values]), axis=0)
    delta_y = (y_max-y_min)
    y_min = y_min - delta_y*0.01
    y_max = y_max + delta_y*0.01
    order = np.argsort(-importance)[:top_k]

    if len(order) < n_cols:
        n_cols = len(order)
        n_rows = 1
    else:
        n_rows = ceil(len(order) / n_cols)
    _, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for iter, i in enumerate(order):
        # Get axis to plot
        curr_ax = get_curr_axis(n_rows, n_cols, ax, iter)

        # To which column of X it corresponds
        column = Imap_inv[i]
        x_min = min(f[:, column].min() for f in foreground)
        x_max = max(f[:, column].max() for f in foreground)
        feature = features.feature_objs[idxs[i]]

        # For each group plot the anchored components in color
        for r in range(n_regions):
            regional_foreground = foreground[r][:, column]
            sorted_idx = np.argsort(regional_foreground)
            regional_foreground = regional_foreground[sorted_idx]
            # Fetch regional interventional decomposition H and Shap values
            H = decomposition[r][(idxs[i],)][sorted_idx]
            Phis = shapley_values[r][sorted_idx, idxs[i]]

            # For ordinal features, we add a jitter to better see the points
            # and we spread the different background via the variable step
            if feature.type in ["bool", "nominal"]:
                jitter = np.random.uniform(-0.05, 0.05, size=len(H))
                step = 0.1 * (r - (n_regions-1) / 2)
                curr_ax.scatter(regional_foreground+jitter+step, Phis, alpha=0.25, c=colors[r])
                # Enlarge the x-range to acount for jitter
                if r == 0:
                    x_min -= 0.05 + 0.1 * (n_regions-1)/2
                    x_max += 0.05 + 0.1 * (n_regions-1)/2
            else:
                step = 0
                curr_ax.scatter(regional_foreground, Phis, alpha=0.25, c=colors[r])
            
            # Plot the PDP as a line
            curr_ax.plot(regional_foreground+step, H, 'k-', linewidth=3)
            if n_regions > 1:
                curr_ax.plot(regional_foreground+step, H, colors[r], linewidth=2)
    
        # xticks labels for categorical data
        if feature.type in ["bool", "ordinal", "nominal"]:
            if feature.type in ["ordinal", "nominal"]:
                categories = feature.cats
                # Truncate names if too long
                # if len(categories) > 5:
                # categories = [name[:3] for name in categories]
                rotation = 45
            else:
                categories = [False, True]
                rotation = 0
            curr_ax.set_xticks(np.arange(len(categories)), labels=categories, rotation=rotation)
        
        # Set ticks and labels
        curr_ax.grid('on')
        curr_ax.set_xlabel(feature.name)
        if iter%n_cols == 0:
            curr_ax.set_ylabel("Local Feature Attribution")
        curr_ax.set_xlim(x_min, x_max)
        if normalize_y:
            curr_ax.set_ylim(y_min, y_max)
            if not iter%n_cols == 0:
                curr_ax.yaxis.set_ticklabels([])
    # Remove unused axes
    iter += 1
    while iter < n_rows * n_cols:
        get_curr_axis(n_rows, n_cols, ax, iter).set_axis_off()
        iter += 1



def plot_legend(rules, figsize=(5, 0.6), ncol=4):
    # Plot the legend separately
    plt.figure(figsize=figsize)
    for p in range(len(rules)):
        plt.scatter(0, 0, alpha=0.5, c=COLORS[p], label=rules[p])
    plt.legend(loc='center', ncol=ncol, prop={"size": 10}, framealpha=1)
    plt.axis('off')



# Visualize the strongest interactions
def plot_interaction(i, j, background, Phis, features):
    
    feature_i = features.feature_objs[i]
    feature_j = features.feature_objs[j]
    plt.figure()
    if feature_j.type == "ordinal":
        for category_idx, category in enumerate(feature_j.cats):
            idx = background[:, j] == category_idx
            plt.scatter(background[idx, i],
                        Phis[idx, i, j], alpha=0.75, c=COLORS[category_idx],
                        label=f"{feature_j.name}={category}")
        plt.legend()
        plt.xlabel(feature_i.name)
        plt.ylabel("Interaction")
    elif feature_j.type == "bool":
        for value in [False, True]:
            idx = np.isclose(background[:, j], int(value))
            plt.scatter(background[idx, i],
                        Phis[idx, i, j], alpha=0.75, c=COLORS[int(value)],
                        label=f"{feature_j.name}={value}")
        plt.legend()
        plt.xlabel(feature_i.name)
        plt.ylabel("Interaction")
    else:
        plt.scatter(background[:, i],
                    background[:, j], c=2*Phis[:, i, j], 
                    cmap='seismic', alpha=0.75)
        plt.xlabel(feature_i.name)
        plt.ylabel(feature_j.name)
        plt.colorbar()
    if feature_i.type == "ordinal":
        plt.xticks(np.arange(len(feature_i.cats)), feature_i.cats)
    # if features.types[j] == "ordinal":
    #    plt.yticks(np.arange(len(features.maps[j].cats)),
    #                features.maps[j].cats)


def interactions_heatmap(Phis, features_names, threshold=0.0005):
    d = len(features_names)
    # We normalize by the model variance
    h_var = Phis.sum(-1).sum(-1).var()
    Phi_imp = (Phis**2).mean(0) / h_var
    np.fill_diagonal(Phi_imp, 0)
    Phi_imp[Phi_imp < threshold] = 0

    fig, ax = plt.subplots(figsize=(d, d))
    im = ax.imshow(Phi_imp, cmap='Reds')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(d))
    ax.set_xticklabels(features_names)
    ax.set_yticks(np.arange(d))
    ax.set_yticklabels(features_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                                    rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(d):
        for j in range(d):
            if Phi_imp[i, j] > 0:
                ax.text(j, i, f"{Phi_imp[i, j]:.3f}",
                        ha="center", va="center", color="w", size=15)

    # ax.set_title("Shapley-Taylor Global indices")
    fig.tight_layout()



def decomposition_graph(decomposition, feature_names):
    
    def interpolate_rgb(rgb_list_1, rgb_list_2, interp):
        """ Linear interpolation interp * rbg_1 + (1 - interp) * rbg_2 """
        out = ''
        for color in range(3):
            hex_color = hex( round(interp * rgb_list_1[color] + \
                                   (1 - interp) * rgb_list_2[color]) )[2:]
            if len(hex_color) == 1:
                hex_color = '0' + hex_color
            out += hex_color
        return out

    # Directed Graph of partial ordering
    dot = Digraph(comment='Functional Decomposition', graph_attr={'ranksep': "0.75"},
                    node_attr={'shape': 'rectangle', 'color': 'black', 'style': 'filled'})
    U = [key for key in decomposition.keys() if len(key)>0]
    n_ranks = max([len(u) for u in U])
    ref_var = np.var(decomposition[()])
    if decomposition[(0,)].ndim == 2:
        var = {u : 100 * np.mean(decomposition[u].mean(1)**2) / ref_var for u in U}
    else:
        var = {u : 100 * np.mean(decomposition[u]**2) / ref_var for u in U}
    max_var = max(list(var.values()))
    ranks_set = []
    for _ in range(n_ranks):
        ranks_set.append(set())

    # Add each feature to the right set
    for u in U:
        ranks_set[ len(u)-1 ].add(u)
    
    # Print
    my_red = color_dict["DEEL"]["neg"]
    for elements in ranks_set:
        with dot.subgraph() as s:
            s.attr(rank='same')
            # Loop over all features of the same rank
            for u in elements:
                s.node(str(u), f":".join([feature_names[i] for i in u]) + "\n" + \
                               f"var={var[u]:.1f}%",
                        fillcolor=f'#{interpolate_rgb(my_red, [255] * 3, var[u]/max_var)}',
                        fontcolor='black')
                if len(u) > 1:
                    for u_subset in combinations(u, len(u)-1):
                        dot.edge(str(u_subset), str(u))
    
    
    return dot
