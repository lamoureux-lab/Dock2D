DatasetGenerator
================

Load/create shapes from the protein pool and compute interactions for IP (docking pose prediction) and FI (fact-of-interaction) datasets.
Interactions that score below specified decision thresholds are added to their respective dataset. Dataset generation
figures and statistics can be generated and saved to file.

.. autoclass:: Dock2D.DatasetGeneration.DatasetGenerator.DatasetGenerator
   :special-members: __init__
   :members: generate_pool, generate_interactions, plot_energy_distributions, plot_accepted_rejected_shapes, generate_datasets, run_generator
   :undoc-members:
