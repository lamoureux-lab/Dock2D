Models/
=======

Models are split into two methodological categories, BruteForce and Sampling, for both the interaction pose (IP) and fact-of-interaction (FI) dataset tasks.

The two core models modules are Docking and Interaction

The docking module producing docking features and scoring coefficients using an SE(2)-Convolutional Neural Network,
and scores the produced features using :doc:`TorchDockingFFT <../api/Dock2D.Utility/Dock2D.Utility.TorchDockingFFT>`



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ../api/Dock2D.Models/Dock2D.Models.BruteForce/Dock2D.Models.BruteForce.rst
   ../api/Dock2D.Models/Dock2D.Models.Sampling/Dock2D.Models.Sampling.rst

.. autoclass:: Dock2D.Models.model_docking.Docking
   :special-members: __init__, forward
   :undoc-members:

.. autoclass:: Dock2D.Models.model_interaction.Interaction
   :special-members: __init__, forward
   :undoc-members:
