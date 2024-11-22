from setuptools import setup, Extension

TREEPATH = "pyfd/tree_shap"

setup(
    ext_modules=[Extension('pyfd.treeshap', sources=[f"{TREEPATH}/main.cpp"], 
                           depends=[ f"{TREEPATH}/leaf_treeshap.hpp", f"{TREEPATH}/progressbar.hpp",
                                     f"{TREEPATH}/recursive_treeshap.hpp", f"{TREEPATH}/utils.hpp",
                                     f"{TREEPATH}/waterfall_treeshap.hpp"])]

)
