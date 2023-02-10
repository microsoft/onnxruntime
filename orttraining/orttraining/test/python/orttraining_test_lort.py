# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ask lazy backend to use Pytorch's JIT as
# lazy backend's executor.
from torch._lazy.ts_backend import init as init_ts_backend

init_ts_backend()

# Handle ORT dependencies.
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

# Set up ORT as torch.jit's sub-executor.
C.register_ort_as_torch_jit_executor()

# Make computation deterministic.
torch.manual_seed(42)
ort.set_seed(1)


class TestOrtLazyTensor(unittest.TestCase):
    def test_elementwise_model(self):
        def run_elementwise_model():
            # A function to test.
            def elementwise_model(x):
                w = x.relu()
                y = w * w + 1.5
                z = y + x
                p = z * x
                q = p.relu()
                return q

            def run(fun, device, x):
                x = torch.tensor(x, device=device, dtype=torch.float32).requires_grad_()
                y = fun(x)
                y.sum().backward()
                return x, y, x.grad

            # Baseline.
            x, y, g_x = run(elementwise_model, "cpu", [-1.0, 2.0])
            # ORT result.
            x_new, y_new, g_x_new = run(elementwise_model, "lazy", [-1.0, 2.0])

            torch.testing.assert_close(x.to("lazy"), x_new)
            torch.testing.assert_close(y.to("lazy"), y_new)
            torch.testing.assert_close(g_x.to("lazy"), g_x_new)

        for _ in range(5):
            run_elementwise_model()

    def test_mnist_model(self):
        def run_mnist_model():
            class MNISTModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
                    self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                    self.fc1 = nn.Linear(9216, 128, bias=False)
                    self.fc2 = nn.Linear(128, 10, bias=False)

                def forward(self, x):
                    x = self.conv1(x)
                    x = F.relu(x)
                    x = self.conv2(x)
                    x = F.relu(x)
                    x = F.max_pool2d(x, 2)
                    x = torch.flatten(x, 1)
                    x = self.fc1(x)
                    x = F.relu(x)
                    x = self.fc2(x)
                    output = F.log_softmax(x, dim=1)
                    return output

            def run(model, device, x, y):
                for param in model.parameters():
                    param.grad = None
                model.to(device)
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                loss = F.nll_loss(output, y)
                # return loss
                loss.backward()
                return loss, (param.grad for param in model.parameters())

            x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
            y = torch.randint(0, 9, (64,), dtype=torch.int64)
            model = MNISTModel()

            # Baseline.
            loss, grads = run(model, "cpu", x, y)
            # ORT result.
            loss_new, grads_new = run(model, "lazy", x, y)

            print(f"MNIST loss: {loss} (pytorch), {loss_new} (ort).")
            torch.testing.assert_close(loss.to("lazy"), loss_new, rtol=1e-2, atol=1e-5)
            for g, g_new in zip(grads, grads_new):
                torch.testing.assert_close(g.to("lazy"), g_new)

        for _ in range(5):
            run_mnist_model()


if __name__ == "__main__":
    # For a specific model, the first 1 or 2 runs of Pytorch
    # JIT is actual eager mode. As a Pytorch JIT sub-executor,
    # ORT won't be unless we run multiple times. Thus, in each
    # test function, we repeat their core test function multiple times.
    # Here we repeat 5 times because we want to our test similar to
    # training loop.
    # TODO: we should force torch.jit executor to use ORT at the first run.
    unittest.main()
