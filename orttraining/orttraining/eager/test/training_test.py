# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# pylint: disable=missing-docstring

import unittest

import onnxruntime_pybind11_state as torch_ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())

test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 5


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


class OrtOpTests(unittest.TestCase):
    """test cases for supported eager ops"""

    def get_device(self):
        return torch_ort.device()

    def test_resize(self):
        device = self.get_device()

        sizes = [[1], [1, 1], [2, 2], [1, 4]]

        # Basic resize from empty Tensor
        for size in sizes:
            torch_size = torch.Size(size)
            cpu_tensor = torch.tensor([])
            ort_tensor = torch.tensor([]).to(device)

            cpu_tensor.resize_(torch_size)
            ort_tensor.resize_(torch_size)

            self.assertEqual(cpu_tensor.size(), ort_tensor.size())

        # Validate cases where we resize from a non-empty tensor
        # to a larger tensor
        cpu_tensor = torch.tensor([1.0, 2.0])
        ort_tensor = cpu_tensor.to(device)

        cpu_tensor.resize_(torch.Size([3]))
        ort_tensor.resize_(torch.Size([3]))

        self.assertEqual(cpu_tensor.size(), ort_tensor.size())
        self.assertTrue(torch.allclose(cpu_tensor[:2], ort_tensor.cpu()[:2]))

        # Validate case when calling resize with current shape & size
        cpu_tensor = torch.tensor([1.0, 2.0])
        ort_tensor = cpu_tensor.to(device)

        cpu_tensor.resize_(torch.Size([2]))
        ort_tensor.resize_(torch.Size([2]))

        self.assertEqual(cpu_tensor.size(), ort_tensor.size())
        self.assertTrue(torch.allclose(cpu_tensor, ort_tensor.cpu()))

        # Validate case when calling resize with different shape but same size
        cpu_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ort_tensor = cpu_tensor.to(device)

        cpu_tensor.resize_(torch.Size([1, 4]))
        ort_tensor.resize_(torch.Size([1, 4]))

        self.assertEqual(cpu_tensor.size(), ort_tensor.size())
        self.assertTrue(torch.allclose(cpu_tensor, ort_tensor.cpu()))

        # Validate cases where we resize from a non-empty tensor
        # to a smaller tensor
        cpu_tensor = torch.tensor([1.0, 2.0])
        ort_tensor = cpu_tensor.to(device)

        cpu_tensor.resize_(torch.Size([1]))
        ort_tensor.resize_(torch.Size([1]))

        self.assertEqual(cpu_tensor.size(), ort_tensor.size())
        self.assertTrue(torch.allclose(cpu_tensor, ort_tensor.cpu()))

    def test_abs_out(self):
        device = self.get_device()
        cpu_tensor = torch.tensor([-1, -2, 3, -6, -7])
        ort_tensor = cpu_tensor.to(device)

        cpu_out_tensor = torch.tensor([], dtype=torch.long)
        ort_out_tensor = cpu_out_tensor.to(device)

        cpu_result = torch.abs(cpu_tensor, out=cpu_out_tensor)
        ort_result = torch.abs(ort_tensor, out=ort_out_tensor)

        assert torch.equal(cpu_result, ort_result.cpu())
        assert torch.equal(cpu_out_tensor, ort_out_tensor.cpu())
        assert torch.equal(ort_result.cpu(), ort_out_tensor.cpu())

    def test_eq_tensor(self):
        device = self.get_device()
        cpu_a = torch.Tensor([1.0, 1.5, 2.0])
        ort_a = cpu_a.to(device)
        cpu_b = torch.Tensor([1.0, 1.5, 2.1])
        ort_b = cpu_b.to(device)

        for tensor_type in {torch.float, torch.bool}:
            for func in {"eq", "ne"}:
                print(f"Testing {func} with type {tensor_type}")
                cpu_out_tensor = torch.tensor([], dtype=tensor_type)
                ort_out_tensor = cpu_out_tensor.to(device)
                cpu_a_b_eq_result = eval(
                    compile("torch." + func + "(cpu_a, cpu_b, out=cpu_out_tensor)", "<string>", "eval")
                )
                ort_a_b_eq_result = eval(
                    compile("torch." + func + "(ort_a, ort_b, out=ort_out_tensor)", "<string>", "eval")
                )
                assert torch.equal(cpu_a_b_eq_result.to(device), ort_a_b_eq_result)
                assert torch.equal(cpu_out_tensor, ort_out_tensor.to("cpu"))
                assert ort_out_tensor.dtype == tensor_type

    def test_eq_scalar(self):
        device = self.get_device()
        cpu_tensor_int = torch.tensor([1, 1], dtype=torch.int32)
        cpu_scalar_int = torch.scalar_tensor(1, dtype=torch.int)
        cpu_scalar_int_not = torch.scalar_tensor(2, dtype=torch.int)
        cpu_tensor_float = torch.tensor([1.1, 1.1], dtype=torch.float32)
        cpu_scalar_float = torch.scalar_tensor(1.1, dtype=torch.float32)
        cpu_scalar_float_not = torch.scalar_tensor(1.0, dtype=torch.float32)

        ort_tensor_int = cpu_tensor_int.to(device)
        ort_scalar_int = cpu_scalar_int.to(device)
        ort_scalar_int_not = cpu_scalar_int_not.to(device)
        ort_tensor_float = cpu_tensor_float.to(device)
        ort_scalar_float = cpu_scalar_float.to(device)
        ort_scalar_float_not = cpu_scalar_float_not.to(device)

        # compare int to int, float to float - ort only supports same type at the moment
        cpu_out_tensor = torch.tensor([], dtype=torch.bool)
        ort_out_tensor = cpu_out_tensor.to(device)

        for func in {"eq", "ne"}:
            cpu_int_int_result = eval(
                compile("torch." + func + "(cpu_tensor_int, cpu_scalar_int, out=cpu_out_tensor)", "<string>", "eval")
            )
            cpu_int_int_not_result = eval(
                compile("torch." + func + "(cpu_tensor_int, cpu_scalar_int_not)", "<string>", "eval")
            )
            cpu_float_float_result = eval(
                compile("torch." + func + "(cpu_tensor_float, cpu_scalar_float)", "<string>", "eval")
            )
            cpu_float_float_not_result = eval(
                compile("torch." + func + "(cpu_tensor_float, cpu_scalar_float_not)", "<string>", "eval")
            )

            ort_int_int_result = eval(
                compile("torch." + func + "(ort_tensor_int, ort_scalar_int, out=ort_out_tensor)", "<string>", "eval")
            )
            ort_int_int_not_result = eval(
                compile("torch." + func + "(ort_tensor_int, ort_scalar_int_not)", "<string>", "eval")
            )
            ort_float_float_result = eval(
                compile("torch." + func + "(ort_tensor_float, ort_scalar_float)", "<string>", "eval")
            )
            ort_float_float_not_result = eval(
                compile("torch." + func + "(ort_tensor_float, ort_scalar_float_not)", "<string>", "eval")
            )

            assert torch.equal(cpu_out_tensor, ort_out_tensor.to("cpu"))
            assert torch.equal(cpu_int_int_result, ort_int_int_result.to("cpu"))
            assert torch.equal(cpu_int_int_not_result, ort_int_int_not_result.to("cpu"))
            assert torch.equal(cpu_float_float_result, ort_float_float_result.to("cpu"))
            assert torch.equal(cpu_float_float_not_result, ort_float_float_not_result.to("cpu"))

    def test_fill(self):
        device = self.get_device()
        for torch_type in [torch.int, torch.float]:
            cpu_tensor = torch.zeros(2, 2, dtype=torch_type)
            ort_tensor = cpu_tensor.to(device)
            for value in [True, 1.1, -1, 0]:
                cpu_tensor.fill_(value)
                ort_tensor.fill_(value)
                assert cpu_tensor.dtype == ort_tensor.dtype
                assert torch.equal(cpu_tensor, ort_tensor.to("cpu"))

    # tests both nonzero and nonzero.out
    def test_nonzero(self):
        device = self.get_device()

        for cpu_tensor in [
            torch.tensor([[[-1, 0, 1], [0, 1, 0]], [[0, 1, 0], [-1, 0, 1]]], dtype=torch.long),
            torch.tensor([[[-1, 0, 1], [0, 1, 0]], [[0, 1, 0], [-1, 0, 1]]], dtype=torch.float),
        ]:
            ort_tensor = cpu_tensor.to(device)

            cpu_out_tensor = torch.tensor([], dtype=torch.long)
            ort_out_tensor = cpu_out_tensor.to(device)

            # nonzero.out
            cpu_result = torch.nonzero(cpu_tensor, out=cpu_out_tensor)
            ort_result = torch.nonzero(ort_tensor, out=ort_out_tensor)
            assert torch.equal(cpu_out_tensor, ort_out_tensor.to("cpu"))
            assert torch.equal(cpu_result, ort_result.to("cpu"))

            # nonzero
            cpu_result = torch.nonzero(cpu_tensor)
            ort_result = torch.nonzero(ort_tensor)
            assert torch.equal(cpu_result, ort_result.to("cpu"))

            # check result between nonzero.out and nonzero
            assert torch.equal(ort_result.to("cpu"), ort_out_tensor.to("cpu"))


if __name__ == "__main__":
    unittest.main()
