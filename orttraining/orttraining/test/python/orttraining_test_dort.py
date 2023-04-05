# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from onnxruntime.training.torchdynamo.register_backend import aot_ort, ort


class TestTorchDynamoOrt(unittest.TestCase):
    """Containers of tests for TorchDynamo ORT (DORT) backend."""

    def setUp(self):
        # Make computation deterministic.
        torch.manual_seed(42)

    def test_elementwise_model(self):
        torch._dynamo.reset()
        """Test DORT with a pure function."""

        def run_elementwise_model():
            # A function to test DORT.
            def elementwise_model(tensor_x: torch.Tensor):
                tensor_w = tensor_x.relu()
                tensor_y = tensor_w * tensor_w + 1.5
                tensor_z = tensor_y + tensor_x
                tensor_p = tensor_z * tensor_x
                tensor_q = tensor_p.relu()
                return tensor_q

            @torch._dynamo.optimize(aot_ort)
            def optimized_elementwise_model(tensor_x: torch.Tensor):
                return elementwise_model(tensor_x)

            def run(fun, list_x):
                tensor_x = torch.tensor(list_x, dtype=torch.float32).requires_grad_()
                tensor_y = fun(tensor_x)
                tensor_y.sum().backward()
                return tensor_x, tensor_y, tensor_x.grad

            # Baseline.
            tensor_x, tensor_y, tensor_x_grad = run(elementwise_model, [-1.0, 2.0])
            # ORT result.
            tensor_x_new, tensor_y_new, tensor_x_grad_new = run(optimized_elementwise_model, [-1.0, 2.0])

            torch.testing.assert_close(tensor_x, tensor_x_new)
            torch.testing.assert_close(tensor_y, tensor_y_new)
            torch.testing.assert_close(tensor_x_grad, tensor_x_grad_new)

        # Run 5 times because ORT runs have side effects and we want to make sure
        # the code is correct with them.
        for _ in range(5):
            run_elementwise_model()

    def test_elementwise_model_for_inference(self):
        torch._dynamo.reset()

        # A function to test DORT for inference (i.e., the compiled function
        # doesn't have backward pass).
        def elementwise_model(tensor_x: torch.Tensor):
            tensor_w = tensor_x.relu()
            tensor_y = tensor_w * tensor_w + 1.5
            tensor_z = tensor_y + tensor_x
            tensor_p = tensor_z * tensor_x
            tensor_q = tensor_p.relu()
            return tensor_q

        @torch._dynamo.optimize(ort)
        def optimized_elementwise_model(tensor_x: torch.Tensor):
            return elementwise_model(tensor_x)

        def run(fun, list_x):
            tensor_x = torch.tensor(list_x, dtype=torch.float32).requires_grad_()
            tensor_y = fun(tensor_x)
            return tensor_y

        # Baseline.
        tensor_y = run(elementwise_model, [-1.0, 2.0])
        # ORT result.
        tensor_y_new = run(optimized_elementwise_model, [-1.0, 2.0])

        torch.testing.assert_close(tensor_y, tensor_y_new)

    def test_to_copy(self):
        torch._dynamo.reset()
        """Test DORT with aten::_to_copy."""

        def run_to_copy():
            # A function to test.
            def copy_copy_copy(tensor_x: torch.Tensor):
                tensor_x1 = torch.ops.aten._to_copy(tensor_x, dtype=torch.int64)
                tensor_x2 = torch.ops.aten._to_copy(tensor_x, dtype=torch.int64, device=tensor_x.device)
                tensor_x3 = torch.ops.aten._to_copy(
                    tensor_x, dtype=torch.int64, device=tensor_x.device, layout=torch.strided
                )
                return tensor_x1, tensor_x2, tensor_x3

            @torch._dynamo.optimize(aot_ort)
            def optimized_copy_copy_copy(tensor_x: torch.Tensor):
                return copy_copy_copy(tensor_x)

            def run(fun, list_x):
                tensor_x = torch.tensor(list_x, dtype=torch.float32)
                tensor_x1, tensor_x2, tensor_x3 = fun(tensor_x)
                return tensor_x1, tensor_x2, tensor_x3

            # Baseline.
            tensor_x, tensor_y, tensor_z = run(copy_copy_copy, [-1.0, 2.0])
            # ORT result.
            tensor_x_new, tensor_y_new, tensor_z_new = run(optimized_copy_copy_copy, [-1.0, 2.0])

            torch.testing.assert_close(tensor_x, tensor_x_new)
            torch.testing.assert_close(tensor_y, tensor_y_new)
            torch.testing.assert_close(tensor_z, tensor_z_new)

        run_to_copy()

    def test_mnist_model(self):
        torch._dynamo.reset()
        """Test DORT with a simple nn.Module."""

        def run_mnist_model():
            class MNISTModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
                    self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
                    self.fc1 = nn.Linear(9216, 128, bias=False)
                    self.fc2 = nn.Linear(128, 10, bias=False)

                def forward(self, tensor_x: torch.Tensor):
                    tensor_x = self.conv1(tensor_x)
                    tensor_x = F.sigmoid(tensor_x)
                    tensor_x = self.conv2(tensor_x)
                    tensor_x = F.sigmoid(tensor_x)
                    tensor_x = F.max_pool2d(tensor_x, 2)
                    tensor_x = torch.flatten(tensor_x, 1)
                    tensor_x = self.fc1(tensor_x)
                    tensor_x = F.sigmoid(tensor_x)
                    tensor_x = self.fc2(tensor_x)
                    output = F.log_softmax(tensor_x, dim=1)
                    return output

            def run(model, tensor_x, tensor_y):
                tensor_x = tensor_x.detach()
                tensor_y = tensor_y.detach()
                for param in model.parameters():
                    param.grad = None
                output = model(tensor_x)
                loss = F.nll_loss(output, tensor_y)
                # return loss
                loss.backward()
                return loss, (param.grad for param in model.parameters())

            # Input.
            tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
            # Label.
            tensor_y = torch.randint(0, 9, (64,), dtype=torch.int64)
            model = MNISTModel()

            # Baseline.
            loss, grads = run(model, tensor_x, tensor_y)
            # ORT result.
            compiled_model = torch._dynamo.optimize(aot_ort)(model)
            loss_new, grads_new = run(compiled_model, tensor_x, tensor_y)

            print(f"MNIST loss: {loss} (pytorch), {loss_new} (ort).")
            torch.testing.assert_close(loss, loss_new, rtol=1e-2, atol=1e-5)
            for grad, grad_new in zip(grads, grads_new):
                torch.testing.assert_close(grad, grad_new)

        # Run 5 times because ORT runs have side effects and we want to make sure
        # the code is correct with them.
        for _ in range(5):
            run_mnist_model()


if __name__ == "__main__":
    unittest.main()
