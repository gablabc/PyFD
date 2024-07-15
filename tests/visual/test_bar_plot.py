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
plt.savefig(os.path.join("Images", f"1_bar_positive.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names)
plt.savefig(os.path.join("Images", f"2_bar_negative.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, absolute=True)
plt.savefig(os.path.join("Images", f"3_bar_negative_abs.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names)
plt.savefig(os.path.join("Images", f"4_bar_mixed.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, absolute=True)
plt.savefig(os.path.join("Images", f"5_bar_mixed_abs.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4], [1.5, -2.3, -3.7, 1]]), features_names[:4])
plt.savefig(os.path.join("Images", f"6_two_bars.png"), bbox_inches='tight')


#### With symetric error bars ####
xerr = np.random.uniform(0.5, 1.5, size=(8,))
bar(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"7_bar_positive_symxerr.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"8_bar_negative_symxerr.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"9_bar_negative_abs_symxerr.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"10_bar_mixed_symxerr.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"11_bar_mixed_abs_symxerr.png"), bbox_inches='tight')

xerr2 = np.random.uniform(0.05, 0.2, size=(8,))
bar(np.array([[1, -2, -3, 4], [1.5, -2.3, -3.7, 1]]), features_names[:4], xerr=[xerr[:4], xerr2[:4]])
plt.savefig(os.path.join("Images", f"12_two_bars_symxerr.png"), bbox_inches='tight')


#### With assymetric error bars ####
xerr = np.random.uniform(0.5, 1.5, size=(2, 8))
bar(np.array([[1, 2, 3, 4, 5, 6, 7, 8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"13_bar_positive_asymxerr.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"14_bar_negative_asymxerr.png"), bbox_inches='tight')

bar(np.array([[-1, -2, -3, -4, -5, -6, -7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"15_bar_negative_abs_asymxerr.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, xerr=xerr)
plt.savefig(os.path.join("Images", f"16_bar_mixed_asymxerr.png"), bbox_inches='tight')

bar(np.array([[1, -2, -3, 4, 5, -6, 7, -8]]), features_names, absolute=True, xerr=xerr)
plt.savefig(os.path.join("Images", f"17_bar_mixed_abs_asymxerr.png"), bbox_inches='tight')

xerr2 = np.random.uniform(0.05, 0.2, size=(2, 8))
bar(np.array([[1, -2, -3, 4], [1.5, -2.3, -3.7, 1]]), features_names[:4], xerr=[xerr[:, :4], xerr2[:, :4]])
plt.savefig(os.path.join("Images", f"18_two_bars_asymxerr.png"), bbox_inches='tight')

