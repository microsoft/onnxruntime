# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
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


class MainWithNonTensorInput(nn.Module):
    # Module for testing non-tensor input.
    def __init__(self):
        super(MainWithNonTensorInput, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.941736), requires_grad=True)
        self.a = A()
        self.b = B()
        self.c = C()
        self.d = D()

    def forward(self, x, case):
        if case == 'reverse':
            z = self.alpha * self.a(self.b(self.c(self.d(x))))
        else:
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


class MainWithMultiModuleOutputs(nn.Module):
    # Module with repeated sub-modules and producing
    # multiple outputs.
    def __init__(self):
        super(MainWithMultiModuleOutputs, self).__init__()
        self.layer_list1 = nn.ModuleList([D(), A(), B()])
        self.layer_list2 = nn.ModuleList([C(), B(), D()])

    def forward(self, x):
        y1 = x
        for layer in self.layer_list1:
            y1 = layer(y1)
        y2 = x
        for layer in self.layer_list2:
            y2 = layer(y2)
        return y1, y2


def test_hierarchical_ortmodule():
    def count_ortmodule(module):
        n = 1 if isinstance(module, ORTModule) else 0
        for sub in module._modules.values():
            n = n + count_ortmodule(sub)
        return n

    def call_backward(y):
        if isinstance(y, tuple):
            for ele in y:
                ele.sum().backward()
        else:
            y.sum().backward()

    def call_allclose(y, y_ref):
        assert type(y) == type(y_ref)
        if isinstance(y, Iterable):
            for ele, ele_ref in zip(y, y_ref):
                torch.allclose(ele, ele_ref)
        else:
            torch.allclose(y, y_ref)

    def trial(module_to_wrap, args, expected_num_ortmodule):
        # Run baseline model.
        m = module_to_wrap

        y_ref = m(*args)
        call_backward(y_ref)
        g_ref = []
        for param in m.parameters():
            g_ref.append(param.grad.detach())

        m.zero_grad()

        # Run hierarchical ORTModule model.
        m = HierarchicalORTModule(m)

        y = m(*args)
        call_backward(y)
        g = []
        for param in m.parameters():
            g.append(param.grad.detach())

        # Some sub-modules become ORTModule.
        assert expected_num_ortmodule == count_ortmodule(m)

        call_allclose(y, y_ref)
        call_allclose(g, g_ref)

    num_trials = 4
    for _ in range(num_trials):
        trial(Main(), [torch.rand(2).requires_grad_()], 6)
        trial(MainWithModuleList(), [torch.rand(2).requires_grad_()], 12)
        trial(MainWithMultiModuleOutputs(), [
              torch.rand(2).requires_grad_()], 10)
        trial(MainWithNonTensorInput(), [
              torch.rand(2).requires_grad_(), 'reverse'], 6)
        trial(MainWithNonTensorInput(), [
              torch.rand(2).requires_grad_(), 'normal'], 6)


if __name__ == '__main__':
    test_hierarchical_ortmodule()
