# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# import lazy_tensor_core as lt
# import lazy_tensor_core.core.lazy_model as ltm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._lazy.ts_backend import init as init_ts_backend

init_ts_backend()
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

# Make computation deterministic.
torch.manual_seed(42)
ort.set_seed(1)
# Set up ORT as JIT's sub-executor.
C.register_ort_as_torch_jit_executor()


def run_Simple():
    # A function to test.
    def Foo(x):
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

    # Beseline.
    x, y, g_x = run(Foo, "cpu", [-1.0, 2.0])
    # ORT result.
    x_new, y_new, g_x_new = run(Foo, "lazy", [-1.0, 2.0])

    print("x:", x)
    print("x_new:", x_new)
    print("y:", y)
    print("y_new:", y_new)
    print("Done assert once")
    assert torch.allclose(x.to("lazy"), x_new)
    assert torch.allclose(y.to("lazy"), y_new)
    assert torch.allclose(g_x.to("lazy"), g_x_new)


def test_Simple():
    for _ in range(5):
        run_Simple()


def run_MNIST():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
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
    model = Net()

    # Baseline.
    loss, grads = run(model, "cpu", x, y)
    # loss = run(model, 'cpu', x, y)
    # ORT result.
    loss_new, grads_new = run(model, "lazy", x, y)
    # loss_new = run(model, 'lazy', x, y)

    print("MNIST result {}, {}, ".format(loss, loss_new))
    assert torch.allclose(loss.to("lazy"), loss_new)
    for g, g_new in zip(grads, grads_new):
        # print(g, g_new)
        assert torch.allclose(g.to("lazy"), g_new)


def test_MNIST():
    for _ in range(5):
        run_MNIST()


if __name__ == "__main__":
    # The first run of Pytorch JIT is actual eager mode,
    # so, as a JIT sub-executor, ORT won't be unless we run
    # multiple times. Thus, in each test function, we repeat
    # their core test function multiple times.
    test_Simple()
    test_MNIST()
