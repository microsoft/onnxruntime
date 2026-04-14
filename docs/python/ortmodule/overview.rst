Overview
=========

`onnxruntime-training`'s `ORTModule` offers a high performance training engine for models defined using the `PyTorch` frontend. `ORTModule` is designed to accelerate the training of large models without needing to change either the model definition or the training code.

The aim of `ORTModule` is to provide a drop-in replacement for one or more `torch.nn.Module` objects in a user's `PyTorch` program, and execute the forward and backward passes of those modules using ORT.

As a result, the user will be able to accelerate their training script using ORT,
without having to modify their training loop.

Users will be able to use standard PyTorch debugging techniques for convergence issues, e.g. by probing the computed gradients on the model's parameters.

The following code example illustrates how ORTModule would be used in a user's training script, in the simple case where the entire model can be offloaded to ONNX Runtime:

.. code-block:: python

    from onnxruntime.training import ORTModule

    # Original PyTorch model
    class NeuralNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            ...
        def forward(self, x):
            ...

    model = NeuralNet(input_size=784, hidden_size=500, num_classes=10)
    model = ORTModule(model) # The only change to the original PyTorch script
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Training Loop is unchanged
    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
