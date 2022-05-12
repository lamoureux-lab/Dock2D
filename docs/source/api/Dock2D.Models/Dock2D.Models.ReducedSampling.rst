Reduced Sampling
================

.. automodule:: Dock2D.Models.ReducedSampling
   :no-members:

.. autoclass:: Dock2D.Models.ReducedSampling.model_sampling.Docker
   :special-members: __init__, forward
   :members:
   :undoc-members:

.. autoclass:: Dock2D.Models.ReducedSampling.model_sampling.SamplingModel
   :special-members: __init__, forward
   :members:
   :undoc-members:

.. autoclass:: Dock2D.Models.ReducedSampling.train_brutesimplified_docking.BruteSimplifiedDockingTrainer
   :special-members: __init__
   :members: run_model, train_model, run_epoch, save_checkpoint, load_checkpoint, resume_training_or_not, run_trainer
   :undoc-members:

.. autoclass:: Dock2D.Models.ReducedSampling.train_montecarlo_interaction.EnergyBasedInteractionTrainer
   :special-members: __init__
   :members: run_model, classify, train_model, run_epoch, checkAPR, save_checkpoint, load_checkpoint, resume_training_or_not, run_trainer
   :undoc-members:
