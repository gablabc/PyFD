import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy



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

