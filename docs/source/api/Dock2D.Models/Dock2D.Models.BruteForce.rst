Brute Force
===========

.. automodule:: Dock2D.Models.BruteForce
   :no-members:

.. autoclass:: Dock2D.Models.BruteForce.train_bruteforce_docking.BruteForceDockingTrainer
   :special-members: __init__
   :members: run_model, train_model, run_epoch, save_checkpoint, load_checkpoint, resume_training_or_not, run_trainer
   :undoc-members:

.. autoclass:: Dock2D.Models.BruteForce.train_bruteforce_interaction.BruteForceInteractionTrainer
   :special-members: __init__
   :members: run_model, classify, train_model, run_epoch, checkAPR, freeze_weights, save_checkpoint, load_checkpoint, set_docking_model_state, resume_training_or_not, run_trainer
   :undoc-members:
