.. Dock2D documentation master file, created by
   sphinx-quickstart on Fri May  6 16:28:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dock2D documentation
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/Dock2D.DatasetGeneration
   api/Dock2D.Models
   api/Dock2D.Tests
   api/Dock2D.Utility


Directory Reference
-------------------

The library is structured into five directories to build a toy protein dataset and models to solve the docking (IP) and interaction (FI) tasks for molecular recognition:

* :doc:`Dock2D.DatasetGeneration <api/Dock2D.DatasetGeneration>`     Generate a protein pool to create IP and FI datasets.

* :doc:`Dock2D.Models <api/Dock2D.Models>`     Four models, two :doc:`BruteForce <api/Dock2D.Models/Dock2D.Models.BruteForce/Dock2D.Models.BruteForce>` and two :doc:`Sampling <api/Dock2D.Models/Dock2D.Models.Sampling/Dock2D.Models.Sampling>` models for the IP and FI tasks each.

* :doc:`Dock2D.Tests <api/Dock2D.Tests>`     Testing scripts to check dataset, algorithm, and model functionality.

* :doc:`Dock2D.Utility <api/Dock2D.Utility>`     Utility classes ranging from data loading to plotting.


Cite Us
-------

This code is part of the work done in `our paper <https://arxiv.org/abs/>`_ .
Please cite us if you use this code in your own work::

    @article{Dock2D,
        title={{Dock2D: Toy datasets for the molecular recognition problem}},
        author={Bhadra-Lobo, Derevyanko, Lamoureux},
        booktitle={TBA},
        year={2022},
    }

Indices and tables
==================

* :ref:`genindex`
