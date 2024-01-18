import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import combinations
from graphviz import Digraph
from math import ceil


from .fd_trees import GADGET_PDP, PDP_PFI_Tree


def setup_pyplot_font(size=11):
    from matplotlib import rc
    rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':size})
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
    """
    # Force the xerr to be a 2d array
    if xerr.ndim == 1:
        xerr = np.vstack((xerr, xerr))
    FI = np.abs(phi)
    cross_origin = ~( np.sign(phi-xerr[0]) == np.sign(phi+xerr[1]) )
    # Min and Max of CIs
    map_bottom = np.abs(phi - xerr[0])
    map_top = np.abs(phi + xerr[1])
    min_CI  = np.minimum(map_bottom, map_top)
    max_CI  = np.maximum(map_bottom, map_top)
    # Minimum of CIs that cross origin is zero
    min_CI[cross_origin] = 0
    assert (min_CI <= FI).all() and (FI <= max_CI).all() 
    # Transform a CI to xerr
    return np.abs(FI - np.vstack((min_CI, max_CI)))
    



def bar(phis, feature_labels, threshold=None, xerr=None, absolute=False, ax=None):
    """
    Plot Feature Importance/Attribution bars

    Parameters
    ----------
    phis : (n_features,) or (stacks, n_features)`np.array`
        Feature importance to plot as bars. If a `List(np.array)` is provided, or
        the array is 2 dimensional, then bars are placed side-by-side.
    feature_labels : `List(string)`
        Name of the features shown no the y-axis. If a `[List(string), List(string)]`
        is provided, then the labels are shown left and right.
    threshold: `float`, default=None
        Show a threshold of significance
    xerr: `np.array` or `List(np.array)`, default=None
        If not *None*, add horizontal vertical errorbars to the bar tips.
        When a list is provided, the error bars are shown for each stacked bar.
        The values are +/- sizes relative to the data:

        - shape(N,): symmetric +/- values for each bar
        - shape(2, N): Separate - and + values for each bar. First row
            contains the lower errors, the second row contains the upper
            errors.

    absolute `bool`, default=False
        Rank with respect to the absolute value of the importance
    ax : `pyplot.axis`
        Add the plots to a pre-existing axis
    """
    # Check phis
    if type(phis) in [list, tuple]:
        assert type(phis[0]) == np.ndarray, "phis must be a list of np.array"
        phis = np.vstack(phis)
        stacked_bars = len(phis)
    else:
        assert type(phis) == np.ndarray, "phis must be a np.array"
        if phis.ndim == 1:
            phis = phis.reshape((1, -1))
            stacked_bars = 1
        else:
            assert phis.shape[1] == len(feature_labels), "phis must be a (stacked, n_features) array"
            stacked_bars = phis.shape[0]
    
    # Are there multiple feature labels?
    if type(feature_labels[0]) == list:
        num_features = len(feature_labels[0])
        multiple_labels = True
    else:
        num_features = len(feature_labels)
        multiple_labels = False

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
        # plt.gcf().set_size_inches(16, 10)
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
        shift = [ (i+1)*bar_width for i in range(stacked_bars//2)[::-1]] + [0]\
                +[-(i+1)*bar_width for i in range(stacked_bars//2)]
    
    # Get DEEL colors
    colors = deepcopy(color_dict["DEEL"])
    colors['pos'] = np.array(colors['pos'])/255.
    colors['neg'] = np.array(colors['neg'])/255.

    # Error bars
    if xerr is not None:
        if xerr.ndim == 2 and xerr.shape[0] == 2:
            xerr = xerr[:, ordered_features]
        else:
            xerr = xerr[ordered_features]
        
    # Plot the bars with err
    for s in range(stacked_bars):
        ax.barh(
            y_pos+shift[s], bar_mapper(phis[s, ordered_features]),
            bar_width, xerr=xerr, align='center',
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
    for i in range(num_features):
        ax.axhline(i, color="k", lw=0.5, alpha=0.5, zorder=-1)

    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    
    if negative_phis:
        ax.set_xlim(xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05)
    else:
        ax.set_xlim(xmin, xmax + (xmax-xmin)*0.05)
    
    plt.gcf().tight_layout()




def partial_dependence_plot(decomposition, foreground, background, features, Imap_inv, normalize_y=True, grouping=None, 
                            figsize=(24, 10), n_cols=5, plot_hist=False, fd_trees_kwargs={}):
    Imap_inv = deepcopy(Imap_inv)
    for i in range(len(Imap_inv)):
        assert len(Imap_inv[i]) == 1, "No feature grouping in PDP plots"
        Imap_inv[i] = Imap_inv[i][0]
    
    anchored = decomposition[()].shape == (foreground.shape[0],)
    additive_keys = [key for key in decomposition.keys() if len(key)==1]
    d = len(additive_keys)

    if anchored:
        y_min = min([np.percentile(h, 15) for h in decomposition.values()])
        y_max = max([np.percentile(h, 85) for h in decomposition.values()])
    else:
        y_min = min([h.min() for h in decomposition.values()])
        y_max = max([h.max() for h in decomposition.values()])
    delta_y = (y_max-y_min)
    y_min = y_min - delta_y*0.01
    y_max = y_max + delta_y*0.01

    n_rows = ceil(d / n_cols)
    _, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, key in enumerate(additive_keys):
        curr_ax = ax[i//n_cols][i%n_cols]
        x_min = foreground[:, Imap_inv[i]].min()
        x_max = foreground[:, Imap_inv[i]].max()
        # delta_x = (x_max-x_min)
        # x_min = x_min - delta_x*0.05
        # x_max = x_max + delta_x*0.05

        if plot_hist:
            _, _, rects = curr_ax.hist(foreground[:, Imap_inv[i]], bins=20, rwidth=0.95, color="gray", alpha=0.6, bottom=y_min)
            max_height = max([h.get_height() for h in rects])
            target_max_height = 0.5 * delta_y
            for r in rects:
                r.set_height(target_max_height*r.get_height()/max_height)
        
        if not anchored:
            sorted_idx = np.argsort(foreground[:, Imap_inv[i]])
            curr_ax.plot(foreground[sorted_idx, Imap_inv[i]], decomposition[key][sorted_idx], 'k-')
        else:
            if grouping is None:
                n_groups = 1
                groups = np.zeros(foreground.shape[1])
            else:
                assert id(foreground) ==  id(background) ,"grouping requires background=foreground"
                decomp_copy = {}
                decomp_copy[()] = decomposition[()]
                decomp_copy[(1,)] = decomposition[key]
                background_slice = np.delete(background, Imap_inv[i], axis=1)
                features_slice = features.remove([Imap_inv[i]])
                fd_tree = GADGET_PDP(features=features_slice, **fd_trees_kwargs) if grouping == "gadget-pdp" \
                            else  PDP_PFI_Tree(features=features_slice, **fd_trees_kwargs) 
                fd_tree.fit(background_slice, decomp_copy)
                groups, rules = fd_tree.predict(background_slice, latex_rules=True)
                n_groups = fd_tree.n_groups

            for group_id in range(n_groups):
                group_foreground = foreground[groups==group_id]
                sorted_idx = np.argsort(group_foreground[:, Imap_inv[i]])
                select = np.where(groups==group_id)[0][sorted_idx].reshape((-1, 1))
                group_ice = decomposition[key][select, select.T]
                # group_ice += mu[groups==group_id].mean()
                curr_ax.plot(group_foreground[sorted_idx, Imap_inv[i]], group_ice, COLORS[group_id], alpha=0.01)
                curr_ax.plot(group_foreground[sorted_idx, Imap_inv[i]], group_ice.mean(1), COLORS[group_id], label=rules[group_id])
        
            if groups is not None:
                curr_ax.legend(fontsize=12, framealpha=1)
        
        # xticks labels depend on the type of feature
        if features.types[Imap_inv[i]] in ["bool", "ordinal"]:
            if features.types[i] == "ordinal":
                categories = features.maps_[i].cats
                # Truncate names if too long
                if len(categories) > 5:
                    categories = [name[0] for name in categories]
            else:
                categories = [False, True]
            curr_ax.set_xticks(np.arange(len(categories)), labels=categories)
                
        curr_ax.grid('on')
        curr_ax.set_xlabel(features.names_[Imap_inv[i]])
        if i%n_cols == 0:
            curr_ax.set_ylabel("Local Feature Attribution")
        curr_ax.set_xlim(x_min, x_max)
        if normalize_y:
            curr_ax.set_ylim(y_min, y_max)
            if not i%n_cols == 0:
                curr_ax.yaxis.set_ticklabels([])



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
    var = {u : 100 * np.mean(decomposition[u].mean(1)**2) / ref_var for u in U}
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