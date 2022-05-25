ValidationMetrics
=====================

Contains classes for evaluation of models performance on both docking tasks.

RMSD for interaction pose (IP) prediction.
Accuracy, precision, recall, F1-score, and MCC (Matthews Correlation Coefficient) for fact-of-interaction (FI).

.. automodule:: Dock2D.Utility.ValidationMetrics
   :undoc-members:

.. autoclass:: Dock2D.Utility.ValidationMetrics.RMSD
   :special-members: __init__
   :members: get_XC, calc_rmsd
   :undoc-members:

.. autoclass:: Dock2D.Utility.ValidationMetrics.APR
   :special-members: __init__
   :members: calc_APR
   :undoc-members:
