import numpy as np
from training_module import TrainingModule
from training_optimizer import TrainingOptimizer

from onnxruntime.capi.onnxruntime_inference_collection import OrtValue

# Random data to test the training loop
data = np.random.rand(10,2)


train_model_uri = "train_model.onnx"
ckpt_uri = "checkpoint.ckpt"
optimizer_model_uri = "optimizer.onnx"


# Create a training module.
model = TrainingModule(train_model_uri, ckpt_uri)

# Create a training optimizer.
optimizer = TrainingOptimizer(optimizer_model_uri, model.parameters())

loss = 0
batch = 0
for epoch in range(10):  # loop over the dataset multiple times
    for i, (x, y) in enumerate(data, 0):

        # get the inputs
        input = OrtValue.ortvalue_from_numpy(x, "cuda", 0)

        fetches = OrtValue.ortvalue_from_numpy(y, "cuda", 0)

        # run forward and backward (training)
        model.train(input, fetches)

        # adjust parameters based on the calculated gradients
        optimizer.step()

        loss += fetches[0].get_tensor_value()
        batch += 1

    # print loss from fetches object.
    print("epoch: {} test loss: {:.3f} ".format(epoch + 1, (loss/batch)))
