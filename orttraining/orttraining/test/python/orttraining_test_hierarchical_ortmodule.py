#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from onnxruntime.training.ortmodule import ORTModule
from onnxruntime.training.ortmodule.experimental.hierarchical_ortmodule import HierarchicalORTModule


class A(nn.Module):
    # A supported module.
    def __init__(self):
        super(A, self).__init__()
        self.l1 = nn.Linear(2, 2)

    def forward(self, x):
        return self.l1(x)


class B(nn.Module):
    # This module is not exportable to ONNX because it
    # uses gradient-checkpointing. However, its two sub-module's
    # are exportable, so ORTModule should be used to compute them.
    def __init__(self):
        super(B, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.a = A()

    def forward(self, x):
        def custom():
            def custom_forward(x_):
                return self.a(x_)
            return custom_forward
        z = self.l1(checkpoint(custom(), x))
        return z


class C(nn.Module):
    # A supported module.
    def __init__(self):
        super(C, self).__init__()
        self.l1 = nn.Linear(2, 2)

    def forward(self, x):
        y = F.relu(self.l1(x))
        return y


class D(nn.Module):
    # This module is not exportable to ONNX because it
    # inner module self.b uses gradient-checkpointing.
    def __init__(self):
        super(D, self).__init__()
        self.b = B()

    def forward(self, x):
        y = F.sigmoid(self.b(x))
        return y


class Main(nn.Module):
    # Main module.
    def __init__(self):
        super(Main, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.941736), requires_grad=True)
        self.a = A()
        self.b = B()
        self.c = C()
        self.d = D()

    def forward(self, x):
        z = self.alpha * self.d(self.c(self.b(self.a(x))))
        return z


class E(nn.Module):
    # Sub-modules are stored in nn.ModuleList.
    def __init__(self):
        super(E, self).__init__()
        self.my_layers = nn.ModuleList([A(), B(), C(), D()])

    def forward(self, x):
        y = x
        for layer in self.my_layers:
            y = layer(y)
        return y


class MainWithModuleList(nn.Module):
    # Sub-modules are stored in nn.ModuleList.
    def __init__(self):
        super(MainWithModuleList, self).__init__()
        self.my_layers = nn.ModuleList([E(), E()])

    def forward(self, x):
        y = x
        for layer in self.my_layers:
            y = layer(y)
        return y


def test_hierarchical_ortmodule():
    def count_ortmodule(module):
        n = 1 if isinstance(module, ORTModule) else 0
        for sub in module._modules.values():
            n = n + count_ortmodule(sub)
        return n

    def trial(module_to_wrap, expected_num_ortmodule):
        x = torch.rand(2).requires_grad_()
        m = module_to_wrap

        y_ref = m(x)
        y_ref.sum().backward()
        g_ref = x.grad.detach()

        x.grad = None
        m = HierarchicalORTModule(m)

        y = m(x,)
        y.sum().backward()
        g = x.grad.detach()

        # Some sub-modules become ORTModule.
        assert expected_num_ortmodule == count_ortmodule(m)

        assert torch.allclose(y, y_ref)
        assert torch.allclose(g, g_ref)

    num_trials = 8
    for i in range(num_trials):
        trial(Main(), 6)
        trial(MainWithModuleList(), 12)


if __name__ == '__main__':
    test_hierarchical_ortmodule()
