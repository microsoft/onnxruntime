Train the Model on the Device
==============================

Once the training artifacts are generated, the model can be trained on the device using the onnxruntime training python API.

The expected training artifacts are:

1. The training onnx model
2. The checkpoint state
3. The optimizer onnx model
4. The eval onnx model (optional)

Sample usage:

.. code-block:: python

    from onnxruntime.training.api import CheckpointState, Module, Optimizer

    # Load the checkpoint state
    state = CheckpointState.load_checkpoint(path_to_the_checkpoint_artifact)

    # Create the module
    module = Module(path_to_the_training_model,
                    state,
                    path_to_the_eval_model,
                    device="cpu")

    optimizer = Optimizer(path_to_the_optimizer_model, module)

    # Training loop
    for ...:
        module.train()
        training_loss = module(...)
        optimizer.step()
        module.lazy_reset_grad()

    # Eval
    module.eval()
    eval_loss = module(...)

    # Save the checkpoint
    CheckpointState.save_checkpoint(state, path_to_the_checkpoint_artifact)


.. autoclass:: onnxruntime.training.api.CheckpointState
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :special-members: __getitem__, __setitem__, __contains__

.. autoclass:: onnxruntime.training.api.Module
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
    :special-members: __call__

.. autoclass:: onnxruntime.training.api.Optimizer
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:

.. autoclass:: onnxruntime.training.api.LinearLRScheduler
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
