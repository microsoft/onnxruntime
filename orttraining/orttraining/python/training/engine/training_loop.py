import numpy as np
import torch
from training_module import TrainingModule
from training_optimizer import TrainingOptimizer

from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValueVector

# Define model, checkpoint and optimizer paths.
train_model_uri = "data/training_model.onnx"
ckpt_uri = "data/checkpoint.ckpt"
optimizer_model_uri = "data/adamw.onnx"

# Generate random data to test with.
inputs = torch.randn(64, 784).numpy()
labels = torch.randint(high=10, size=(64,), dtype=torch.int32).numpy()

# Create a Training Module and Training Optimizer.
model = TrainingModule(train_model_uri, ckpt_uri)
optimizer = TrainingOptimizer(optimizer_model_uri, model.get_model())

# Create OrtValue Vector to pass it to the model train function.
forward_inputs = OrtValueVector()
forward_inputs.reserve(2)
forward_inputs.push_back(OrtValue.ortvalue_from_numpy(inputs)._ortvalue)
forward_inputs.push_back(OrtValue.ortvalue_from_numpy(labels)._ortvalue)

# Create an Empty OrtValueVector to save the loss value on it.
forward_output = OrtValueVector()


for epoch in range(10):
    model.reset_grad()

    model.train(forward_inputs, forward_output)

    # Run the optimizer step.
    optimizer.step()

    # After running the train step, the forward_output should have a loss value on it.
    print("epoch {}, loss {}".format(epoch, forward_output[0].numpy()))
