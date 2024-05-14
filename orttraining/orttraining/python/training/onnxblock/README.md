## Getting Started

This is a simple guide on how to generate the training graphs using the offline tools.

### What's needed for training?
The ort training APIs need the following files for performing training
1. The training onnx model.
2. The eval onnx model (optional).
3. The optimizer onnx model.
4. The checkpoint file.

The offline tooling helps generating the above files. Read along to know how.

### Generating the artifacts

The starting point to generating the training artifacts is to have a forward only onnx model available. This model will be used as the base model for creating the training model. If the user is converting from PyTorch, it is suggested that the user use the following arguments while exporting the model:
1. export_params: True
2. training: torch.onnx.TrainingMode.TRAINING
3. do_constant_folding: False

An example command for exporting the model can be:

```py
torch.onnx.export(model, sample_inputs, "base_model.onnx",
                  export_params=True, training=torch.onnx.TrainingMode.TRAINING,
                  do_constant_folding=False)
```

Now, onto generating the training artifacts. Let's assume that a forward only onnx model has already been generated. Here, we require that the user's onnx model is generated with the parameters embedded inside the exported model, i.e. with the parameters `export_params=True` and `training=torch.onnx.TrainingMode.TRAINING`.

```py
from onnxruntime.training import artifacts


# Load the onnx model
model_path = "model.onnx"
base_model = onnx.load(model_path)

# Define the parameters that need their gradient computed
requires_grad = ["weight1", "bias1", "weight2", "bias2"]
frozen_params = ["weight3", "bias3"]

# Generate the training artifacts
artifacts.generate_artifacts(base_model, requires_grad = requires_grad, frozen_params = frozen_params,
                             loss = artifacts.LossType.CrossEntropyLoss, optimizer = artifacts.OptimType.AdamW)

# Successful completion of the above call will generate 4 files in the current working directory,
# one for each of the artifacts mentioned above (training_model.onnx, eval_model.onnx, checkpoint, optimizer_model.onnx)
```

Once the models and checkpoint have been generated, they can be loaded in the online training step and executed.
For an example on how the online training loop should be written given these generated files, refer to this
[sample trainer](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/test/training_api/trainer/trainer.cc).

### Advanced scenarios

Let's assume, we want to use a custom loss function with a model. For this example, we assume that our model generates
two outputs. And the custom loss function must apply a loss function on each of the outputs and perform a weighted average
on the output. Mathematically,

```
loss = 0.4 * mse_loss1(output1, target1) + 0.6 * mse_loss2(output2, target2)
```

Since this is a custom loss function, this loss function is not exposed as an enum from `generate_artifacts` function.

For this, we make use of `onnxblock`.

```py
import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training import artifacts

# Define a custom loss block that takes in two inputs
# and performs a weighted average of the losses from these
# two inputs.
class WeightedAverageLoss(onnxblock.Block):
    def __init__(self):
        self._loss1 = onnxblock.loss.MSELoss()
        self._loss2 = onnxblock.loss.MSELoss()
        self._w1 = onnxblock.blocks.Constant(0.4)
        self._w2 = onnxblock.blocks.Constant(0.6)
        self._add = onnxblock.blocks.Add()
        self._mul = onnxblock.blocks.Mul()

    def build(self, loss_input_name1, loss_input_name2):
        # The build method defines how the block should be stacked on top of
        # loss_input_name1 and loss_input_name2

        # Returns weighted average of the two losses
        return self._add(
            self._mul(self._w1(), self._loss1(loss_input_name1, target_name="target1")),
            self._mul(self._w2(), self._loss2(loss_input_name2, target_name="target2"))
        )

my_custom_loss = WeightedAverageLoss()

# Load the onnx model
model_path = "model.onnx"
base_model = onnx.load(model_path)

# Define the parameters that need their gradient computed
requires_grad = ["weight1", "bias1", "weight2", "bias2"]
frozen_params = ["weight3", "bias3"]

# Now, we can invoke generate_artifacts with this custom loss function
artifacts.generate_artifacts(base_model, requires_grad = requires_grad, frozen_params = frozen_params,
                             loss = my_custom_loss, optimizer = artifacts.OptimType.AdamW)

# Successful completion of the above call will generate 4 files in the current working directory,
# one for each of the artifacts mentioned above (training_model.onnx, eval_model.onnx, checkpoint, optimizer_model.onnx)
```

For more advanced scenarios, refer to [onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples)
