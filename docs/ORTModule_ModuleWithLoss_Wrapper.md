# ONNX Runtime ModuleWithLoss Wrapper

This document provides instructions on implementing a wrapper similar to the ModuleWithLoss Wrapper in Optimum. By implementing this wrapper, you can compute the loss inside ONNX Runtime (ORT), enabling you to leverage additional optimizations such as label sparsity optimization.

**Note: The adaptation described below is not necessary for all cases. It is only needed in specific scenarios, which we will clarify below:**

1. When the loss is not computed in the model's forward path.
2. When the model's forward path computes the loss but also returns other outputs that are not needed for subsequent computations.

In the first case, if the loss is not computed in the model's forward pass, ONNX Runtime (ORT) cannot track the loss computation in the ONNX graph.

In the second case, if the model's forward pass computes the loss but also returns additional tensors that are not needed for subsequent computations, using the original model directly with the ORT wrapper can lead to unnecessary memory usage on the CUDA device during backward computations.

## Implementation Steps

Follow these steps to create your own ModuleWithLoss for computing loss inside ONNX Runtime:

Certainly! Here are the steps to fit the provided example:

### Step 1: Define `ModuleWithLoss` Class

1. Create a class named `ModuleWithLoss` that extends `nn.Module`.
2. Implement the `__init__` method to initialize the wrapper with the original model.
3. Implement the `forward` method to perform the forward pass of the model and compute the loss.
4. Use the model's `forward` method to compute the logits.
5. Compute the loss between the logits and the labels.
6. Return the computed loss.

```python
class ModuleWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels):
        # Perform the forward pass of the model
        lm_logits = self.model(inputs)

        # Compute the cross-entropy loss
        loss = nn.CrossEntropyLoss()(lm_logits, labels)
        return loss
```
### Step 2: Define a sample training script

#### Define `PretrainedModel` Class

1. Implement the `PretrainedModel` class by extending `nn.Module`.
2. Define the `forward` method inside `PretrainedModel` to perform the forward pass of the model.
3. Use the model's transformer layers and head layers to compute the logits.
4. Return the logits as the output of the `forward` method.

#### Training Loop

1. Create an instance of `PretrainedModel` as the original model.
2. Create an instance of `ModuleWithLoss` by passing the original model as an argument.
3. Initialize the optimizer for training.
4. Enter the training loop and iterate over the training data.
5. Zero the gradients of the optimizer.
6. Compute the forward pass and cross-entropy loss by calling the `forward` method of the `ModuleWithLoss` instance and passing the inputs and labels.
7. Perform the backward pass by calling `loss.backward()` and optimization step by calling `optimizer.step()`.

Make sure to fill in the appropriate details and customize the code as per your specific requirements and implementation.

```python
# Define the model architecture
class PretrainedModel(nn.Module):
    ...

    def forward(self, input_ids, attention_mask):
        ...
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            ...
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

# Training loop
model = PretrainedModel(...)
model = ModuleWithLoss(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model = ORTModule(model)

for inputs, labels in dataloader:
    optimizer.zero_grad()

    # Compute the forward pass and cross-entropy loss
    loss = model(inputs, labels)

    # Backward pass and optimization step
    loss.backward()
    optimizer.step()

```

By following these steps, you can create a wrapper that computes the loss inside ONNX Runtime (ORT) using the `ModuleWithLoss` class and the `compute_loss` function. Make sure to customize the code snippets according to your specific codebase and requirements.

Please note that the steps provided above are specific to the example I provided, and you may need to adapt them based on your own implementation and requirements.
