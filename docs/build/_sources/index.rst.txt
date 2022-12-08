.. Dock2D documentation master file, created by
   sphinx-quickstart on Fri May  6 16:28:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Dock2D Doc(k)umentation
==================================

The library is used to build a toy protein dataset and models
to solve the tasks for molecular recognition:

   .. glossary::

       Interaction Pose (IP):
           Solve for a specific pose between two proteins known to interact.
           This is the pose generated from the lowest energy transformation that brings a ligand spatially proximal to a receptor.

       Fact-of-Interaction (FI):
           Solve for whether two proteins meaningfully interact, if at all.
           This is the probability of two proteins interacting using the free energy of possible transformations.



Directory Reference
-------------------

The library is structured into five directories:

* :doc:`Dock2D.DatasetGeneration <api/Dock2D.DatasetGeneration>`     Generate a protein pool to create IP and FI datasets.

* :doc:`Dock2D.Models <api/Dock2D.Models>`     Four models, two :doc:`BruteForce <api/Dock2D.Models/Dock2D.Models.BruteForce/Dock2D.Models.BruteForce>` and two :doc:`Sampling <api/Dock2D.Models/Dock2D.Models.Sampling/Dock2D.Models.Sampling>` models for the IP and FI tasks each.

* :doc:`Dock2D.Tests <api/Dock2D.Tests>`     Testing scripts to check dataset, algorithm, and model functionality.

* :doc:`Dock2D.Utility <api/Dock2D.Utility>`     Utility classes ranging from data loading to plotting.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/Dock2D.DatasetGeneration
   api/Dock2D.Models
   api/Dock2D.Tests
   api/Dock2D.Utility


Cite Us
-------

This code is part of the work done in `our paper <https://arxiv.org/abs/2212.03456>`_ .
Please cite us if you use this code in your own work:

.. code-block:: text

    @article{Dock2D,
        title={{Dock2D: Synthetic datasets for the molecular recognition problem}},
        author={Bhadra-Lobo, Derevyanko, Lamoureux},
        journal={arXiv preprint arxiv:2212.03456},
        year={2022},
    }


Indices and tables
------------------

* :ref:`genindex`
