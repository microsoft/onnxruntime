# ONNX Runtime ModuleWithLoss Wrapper

This document provides instructions on implementing a wrapper similar to the ModuleWithLoss Wrapper in Optimum. By implementing this wrapper, you can compute the loss inside ONNX Runtime (ORT), enabling you to leverage additional optimizations such as label sparsity optimization.

#### ModuleWithLoss Wrapper

- **Description**: The ModuleWithLoss wrapper is a class that extends the nn.Module class from PyTorch. It is designed to enhance the functionality of an existing model by allowing the computation of loss inside the wrapper, leveraging the benefits of ONNX Runtime (ORT) and enabling additional optimizations such as label sparsity optimization.

These are the Major changes on the code, Refer to these snippets as a reference while writing your own implementation.

```python
class ModuleWithLoss(nn.Module):
    def __init__(self, model, args, label_smoother):
        super().__init__()
        self._original_model = model
        self.args = args
        # Label smoothing
        self.label_smoother = label_smoother

    def forward(self, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs):
        # The compute_model_plus_loss_internal is assigned once the class is instantiated.
        # It should have same signature as Trainer.compute_loss().
        # We do this to avoid potential un-synced states if we duplicated compute loss codes .
        return self.compute_model_plus_loss_internal(self._original_model, inputs, return_outputs)

    @property
    def module(self):
        """The original `torch.nn.Module` that this module wraps.
        This property provides access to methods and properties on the original module."""

        return self._original_model.module

    @property
    def config(self):
        return self._original_model.config

def create_model_with_loss(self):
    model_with_loss = ModuleWithLoss(self.model, self.args, self.label_smoother)
    model_with_loss.compute_model_plus_loss_internal = types.MethodType(Trainer.compute_loss, model_with_loss)

    return model_with_loss

def _set_signature_columns_if_needed(self):
    if self._signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        import inspect

        if isinstance(self.model, ModuleWithLoss):
            signature = inspect.signature(self.model._original_model.forward)
        else:
            signature = inspect.signature(self.model.forward)

        self._signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

def compute_loss(self, model_with_loss, inputs, return_outputs=False):
    # Run model forward + loss compute.
    if isinstance(self.model, ModuleWithLoss):
        # ORTModule Does not support the BatchEncoding Type so we have to convert to a dict.
        dict_inputs = dict(inputs.items())
        return model_with_loss(dict_inputs, return_outputs)
    else:
        return super().compute_loss(model_with_loss, inputs, return_outputs)

```

## Creating the ModuleWithLoss Wrapper

### 1. Define `ModuleWithLoss` Class

- Create a class named `ModuleWithLoss` that extends `nn.Module`.
- Implement the __init__ method to initialize the wrapper with the original model, training arguments (`args`), and label smoother.
- Add an instance variable to store the original model (`_original_model`).
- Include the label smoothing functionality if required.

### 2. Override the `forward` Method

- In the `ModuleWithLoss` class, override the forward method.
- Within this method, call `compute_model_plus_loss_internal` to compute the loss using the original model, inputs, and return_outputs.
- Return the computed loss.

### 3. Define the `module` and `config` properties

- Implement the `module` property to provide access to methods and properties on the original model.
- Implement the `config` property to return the configuration of the original model.

## Initializing the Wrapper

### 4. Create a function to initialize the wrapper:
- Implement a function, such as `create_model_with_loss()`, to create an instance of `ModuleWithLoss` using the original model, arguments, and label smoother.
- Assign `compute_model_plus_loss_internal` as a method of `Trainer.compute_loss` to the wrapper instance.
- Return the created wrapper instance.

### 5. Override the necessary functions:
- Override the `_set_signature_columns_if_needed` function if needed.
- Update the function to handle the case where the model is an instance of `ModuleWithLoss`.

### 6. Override the `compute_loss` function if needed:
- Override the `compute_loss` function to handle the case where the model is an instance of `ModuleWithLoss`.
- Run the model forward pass and compute the loss.
- Return the computed loss.

## Model Initialization

### 7. Initialize the model in the `__init__` function:
- During model initialization, create the training model instance using `create_model_with_loss()`.

### 8. Update eval and predict methods:
- In the evaluation and prediction methods (`eval` and `predict`), fallback to the `_original_model` of `ModuleWithLoss` to ensure correct behavior.

By following these steps, you can create your own wrapper to put the loss computation inside ONNX Runtime (ORT). Make sure to customize the code snippets according to your specific codebase and requirements.

Please note that the steps provided above are general guidelines, and the specific steps required may vary depending on your codebase. It's possible that you may not need all of these steps, or you may require additional steps based on your specific implementation.
