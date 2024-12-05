Tutorials
=========

Introduction 
------------
These notebooks introduce the general workflow of the ``PyFD`` package : computing functional decomposition, 
aggregating them to get PDP/SHAP/PFI additive explanations, increase their alignment with feature grouping and FD-Trees.

.. toctree::
   :maxdepth: 1

   tutorials/1-1-PyFD-Introduction.ipynb
   tutorials/1-2-Regression-Introduction.ipynb
   tutorials/1-3-Classification-Introduction.ipynb

Explaining Additive Models 
--------------------------
The fundamental goal of ``PyFD`` is to compute faithful explanations for the predictions of complex black-box models. 
Yet there is no agreement on what faithful explanations should look like when investigating general models. 
Consequently, we argue that it is best to first study models that can be faithfully explained : **Additive Models**. 
This series of tutorials show how to fit and interpret additive models. 

.. toctree::
   :maxdepth: 1

   tutorials/2-1-Explain-Linear-Models.ipynb
   tutorials/2-2-Explain-Parametric-Additive-Models.ipynb

Explaining General Models 
-------------------------
Once we agree on how additive models should be explained, we can generalize said definitions to more general models 
by leveraging the theory of **Functional Decomposition**. As this suite of tutorials will highlight, Functional 
Decomposition offers a uniform framework for computing the PDP/SHAP/PFI explainability methods.

**TODO**

Increasing Explanations Alignment
---------------------------------
When explaning models that contain feature interactions (*i.e.* the model is no longer additive), the PDP/SHAP/PFI methods 
will disagree and potentially lead to contradictory models insights. Thus, we advocate minimizing the strength of feature 
interactions to increase the agreement between the various methods. ``PyFD`` offers two methodologies
to minimize interactions : grouping features and FD-Trees, which are both discussed in the notebook series.

.. toctree::
   :maxdepth: 1

   tutorials/4-1-FDTrees.ipynb
   tutorials/4-2-FDTrees.ipynb
