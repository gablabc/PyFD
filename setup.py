from setuptools import setup, Extension

setup(
    ext_modules=[Extension('pyfd.treeshap', sources=['pyfd/tree_shap/main.cpp'],),],
)