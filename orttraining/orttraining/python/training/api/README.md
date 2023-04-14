## Getting Started

This is a simple guide on how to use onnxruntime training APIs.

### What's needed for training?
The ort training APIs need the following files for performing training
1. The training onnx model.
2. The eval onnx model (optional).
3. The optimizer onnx model (optional).
4. The checkpoint file.

To generate these files, refer to this [onnxblock's README](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md)


Once the onnx models are generated, you can use the training APIs to run your training.

### Training Loop

```py
from onnxruntime.training.api import Module, Optimizer, CheckpointState

# Create Checkpoint State.
state = CheckpointState.load_checkpoint("checkpoint.ckpt")

# Create Module and Optimizer.
model = Module("training_model.onnx", state, "eval_model.onnx")
optimizer = Optimizer("optimizer.onnx", model)

# Set model in training mode and run a Train step.
model.train()
training_model_outputs = model(<inputs to your training model>)

# Optimizer step
optimizer.step()

# Set Model in eval mode and run an Eval step.
model.eval()

eval_model_outputs = model(<inputs to your eval model>)

# Assuming that the loss is the first element of the output in the training model.
print("Loss : ", training_model_outputs[0])

# Saving checkpoint.
state = model.get_state()
CheckpointState.save_checkpoint(state, "checkpoint_export.ckpt")

```

For more detailed information refer to [Module](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/api/Module.py) and [Optimizer](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/api/Optimizer.py).
