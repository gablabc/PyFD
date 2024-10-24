# 🧠 Tutorials

This is a series of tutorials meant to explain the principles behind the `PyFD` package, its workflow, and its API.
It consists of four series of jupyter notebooks.

## 1 Introduction
These notebooks introduce the general workflow of the `PyFD` package : computing functional decomposition, aggregating them to
get PDP/SHAP/PFI additive explanations, increase their alignment with feature grouping and FD-Trees.
- [1-1-PyFD-Introduction](https://github.com/gablabc/PyFD/blob/master/tutorials/1-1-PyFD-Introduction.ipynb) illustrates the package on a toy dataset and model.
- [1-2-Regression-Introduction](https://github.com/gablabc/PyFD/blob/master/tutorials/1-2-Regression-Introduction.ipynb) applies `PyFD` on a regression use-case (bike rental forecasting).
- [1-3-Classification-Introduction](https://github.com/gablabc/PyFD/blob/master/tutorials/1-3-Classification-Introduction.ipynb) employs the package on a classification scenario (marketing phone call successes).

## 2 Explaining Additive Models
The fundamental goal of `PyFD` is to compute faithful explanations for the predictions of complex black-box models. Yet there is no agreement on what a faithful explanation
should look like when investigating general models. Consequently, we argue that it is best to first investigate models that can be faithfully explained : **Additive Models**. 
This series of tutorials show how to fit and interpret additive models. 

**TODO**

## 3 Explaining General Models 
Once we agree on how additive models should be explained, we can generalize said definitions to more general models by leveraging the theory of **Functional Decomposition**. 
As this suite of tutorials will highlight, Functional Decomposition offers a uniform framework for computing thePDP/SHAP/PFI explability methods.

**TODO**

## 4 Increasing Explanations Alignment
When explaning models that contain feature interactions (*i.e.* the model is no longer additive), the PDP/SHAP/PFI methods will disagree and potentially lead to contradictory
models insights. Thus, we advocate minimizing the strength of feature interactions to increase the agreement between the various methods. `PyFD` offers two methodologies
to minimize interactions : grouping features and FD-Trees, which are both discussed in the notebook series.

- [4-1-FDTrees](https://github.com/gablabc/PyFD/blob/master/tutorials/4-1-FDTrees.ipynb) introduces FDTrees on the simplest possible example.
- [4-2-FDTrees](https://github.com/gablabc/PyFD/blob/master/tutorials/4-2-FDTrees.ipynb) discusses the mathematics and API of FDTrees in more details. 
