check_param_state_dict_cuda.py
==============================

Run a dummy model to show that model parameters can be saved and loaded from the `state_dict`
This is useful for resuming training, loading specific epochs for evaluation, or loading pretrained models.

.. autoclass:: Dock2D.Tests.check_param_state_dict_cuda.DummyModel
   :special-members: __init__, forward
   :members: save_checkpoint, load_checkpoint
   :undoc-members:
