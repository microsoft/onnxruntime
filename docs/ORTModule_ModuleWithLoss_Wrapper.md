# ONNX Runtime ModuleWithLoss Wrapper

This document provides instructions on implementing a wrapper similar to the ModuleWithLoss Wrapper in Optimum. By implementing this wrapper, you can compute the loss inside ONNX Runtime (ORT), enabling you to leverage additional optimizations such as label sparsity optimization.
## Implementation Steps

Follow these steps to create your own ModuleWithLoss for computing loss inside ONNX Runtime:

### Step 1: Define `ModuleWithLoss` Class

1. Create a class named `ModuleWithLoss` that extends `nn.Module`.
2. Implement the `__init__` method to initialize the wrapper with the original model.
3. Implement the `forward` method to perform the forward pass of the model and return the outputs.

### Step 2: Compute Loss Function

1. Implement a function named `compute_loss` that takes the model, inputs, and targets as arguments.
2. Inside the function, compute the forward pass of the model using the inputs.
3. Compute the loss between the model outputs and the targets.
4. Return the computed loss.

### Step 3: Training Loop

1. Create an instance of `ModuleWithLoss` by passing the original model as an argument.
2. Initialize the optimizer for training.
3. Enter the training loop and iterate over the training data.
4. Zero the gradients of the optimizer.
5. Compute the loss by calling the `compute_loss` function and passing the model, inputs, and targets.
6. Perform the backward pass and optimization step by calling `loss.backward()` and `optimizer.step()`.


## Full Example

```python
class ModuleWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        # Perform the forward pass of the model and return the outputs
        outputs = self.model(inputs)
        return outputs

def compute_loss(model, inputs, targets):
    # Compute the forward pass of the model
    outputs = model(inputs)

    # Compute the cross-entropy loss
    loss = nn.CrossEntropyLoss()(outputs, targets)

    return loss

# Training loop
model = ModuleWithLoss(original_model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for inputs, targets in dataloader:
    optimizer.zero_grad()

    # Compute loss
    loss = compute_loss(model, inputs, targets)

    # Backward pass and optimization step
    loss.backward()
    optimizer.step()
```

By following these steps, you can create a wrapper that computes the loss inside ONNX Runtime (ORT) using the `ModuleWithLoss` class and the `compute_loss` function. Make sure to customize the code snippets according to your specific codebase and requirements.

Please note that the steps provided above are specific to the example I provided, and you may need to adapt them based on your own implementation and requirements.
