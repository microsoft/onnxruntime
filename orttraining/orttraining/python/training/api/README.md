## Getting Started

This is a simple guide on how to use onnxruntime training APIs.

### What's needed for training?
The ort training APIs need the following files for performing training
1. The training onnx model.
2. The eval onnx model (optional).
3. The optimizer onnx model.
4. The checkpoint file.

To generate these files, refer to this [onnxblock's README](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md)


Once the onnx models are generated, you can use the training APIs to run your training.

### Training Loop

```py
from onnxruntime.training.api import Module, Optimizer, CheckpointState
# Create Checkpoint State.
state = CheckpointState("checkpoint.ckpt")
# Create Module and Optimizer.
model = Module("training_model.onnx", state, "eval_model.onnx")
optimizer = Optimizer("optimizer.onnx", model)

# Data should be a list of numpy arrays.
forward_inputs = ...

# Set model in training mode and run a Train step.
model.train()
model(forward_inputs)

# Optimizer step
optimizer.step()

# Set Model in eval mode and run an Eval step.
model.eval()

loss = model(forward_inputs)

# Assuming that the loss is the first element of the output in our case.
print("Loss : ", loss[0])

# Saving checkpoint.
model.save_checkpoint("checkpoint_export.ckpt")

```

For more detailed information refer to [Module](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/api/Module.py) and [Optimizer](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/api/Optimizer.py).
