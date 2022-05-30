Models/
=======

Models are split into two methodological categories, BruteForce and Sampling, for both the interaction pose (IP) and fact-of-interaction (FI) dataset tasks.
For BruteForce, the models sample the entire space of rotations and translations.
For Sampling, only a small subset of the transformational space is sampled.

The two core models modules are :class:`Docking() <Dock2D.Models.model_docking.Docking>` and :class:`Interaction() <Dock2D.Models.model_interaction.Interaction>`.

The docking module produces docking features and scoring coefficients using an SE(2)-Convolutional Neural Network,
and scores the produced features using :doc:`TorchDockingFFT <../api/Dock2D.Utility/Dock2D.Utility.TorchDockingFFT>`.

The interaction module computes a probability of interaction based on free energies learned using the docking module.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ../api/Dock2D.Models/Dock2D.Models.BruteForce/Dock2D.Models.BruteForce.rst
   ../api/Dock2D.Models/Dock2D.Models.Sampling/Dock2D.Models.Sampling.rst
   ../api/Dock2D.Models/Dock2D.Models.TrainerIP.rst
   ../api/Dock2D.Models/Dock2D.Models.TrainerFI.rst

.. autoclass:: Dock2D.Models.model_docking.Docking
   :special-members: __init__, forward
   :undoc-members:

.. autoclass:: Dock2D.Models.model_interaction.Interaction
   :special-members: __init__, forward
   :undoc-members:

.. autoclass:: Dock2D.Models.model_sampling.SamplingDocker
   :special-members: __init__, forward
   :undoc-members:

.. autoclass:: Dock2D.Models.model_sampling.SamplingModel
   :special-members: __init__, forward
   :undoc-members:
