""" Verify the bar plot function on local and global importance """

import numpy as np
import matplotlib.pyplot as plt
import os

from pyfd.plots import bar, setup_pyplot_font

setup_pyplot_font(20)
np.random.seed(1)
features_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

#### Without error bars ####
bar(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), features_names)
plt.savefig(os.path.join("Images", f"bar_positive.pdf"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names)
plt.savefig(os.path.join("Images", f"bar_negative.pdf"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, absolute=True)
plt.savefig(os.path.join("Images", f"bar_negative_abs.pdf"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names)
plt.savefig(os.path.join("Images", f"bar_mixed.pdf"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, absolute=True)
plt.savefig(os.path.join("Images", f"bar_mixed_abs.pdf"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4], [1.5, -2.3, -3.7, 1]]), features_names[:4])
plt.savefig(os.path.join("Images", f"two_bars.pdf"), bbox_inches='tight')

#### With error bars ####
xerr = np.random.uniform(0.5, 1.5, size=(2, 8))
bar(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_positive_xerr.pdf"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_negative_xerr.pdf"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_negative_abs_xerr.pdf"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_mixed_xerr.pdf"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_mixed_abs_xerr.pdf"), bbox_inches='tight')

# xerr2 = np.random.uniform(0.05, 0.2, size=(2, 8))
# bar(np.array([[1, -2, -3, 4], [1.5, -2.3, -3.7, 1]]), features_names[:4], xerr=[xerr[:, :4], xerr2[:, :4]])
# plt.savefig(os.path.join("Images", f"two_bars_xerr.pdf"), bbox_inches='tight')

