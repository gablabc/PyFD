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
plt.savefig(os.path.join("Images", f"bar_1_positive.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names)
plt.savefig(os.path.join("Images", f"bar_2_negative.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, absolute=True)
plt.savefig(os.path.join("Images", f"bar_3_negative_abs.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names)
plt.savefig(os.path.join("Images", f"bar_4_mixed.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, absolute=True)
plt.savefig(os.path.join("Images", f"bar_5_mixed_abs.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4], [1.5, -2.3, -3.7, 1]]), features_names[:4])
plt.savefig(os.path.join("Images", f"bar_6_two.png"), bbox_inches='tight')


#### With symetric error bars ####
xerr = np.random.uniform(0.5, 1.5, size=(8,))
bar(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_7_positive_symxerr.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_8_negative_symxerr.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_9_negative_abs_symxerr.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_10_mixed_symxerr.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_11_mixed_abs_symxerr.png"), bbox_inches='tight')

xerr2 = np.random.uniform(0.05, 0.2, size=(8,))
bar(np.array([[1, -2, -3, 4], [1.5, -2.3, -3.7, 1]]), features_names[:4], xerr=[xerr[:4], xerr2[:4]])
plt.savefig(os.path.join("Images", f"bar_12_two_symxerr.png"), bbox_inches='tight')


#### With assymetric error bars ####
xerr = np.random.uniform(0.5, 1.5, size=(2, 8))
bar(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_13_positive_asymxerr.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_14_negative_asymxerr.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_15_negative_abs_asymxerr.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_16_mixed_asymxerr.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"bar_17_mixed_abs_asymxerr.png"), bbox_inches='tight')

xerr2 = np.random.uniform(0.05, 0.2, size=(2, 8))
bar(np.array([[1, -2, -3, 4], [1.5, -2.3, -3.7, 1]]), features_names[:4], xerr=[xerr[:, :4], xerr2[:, :4]])
plt.savefig(os.path.join("Images", f"bar_18_two_asymxerr.png"), bbox_inches='tight')

