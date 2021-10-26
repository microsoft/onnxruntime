# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import torch


class MyCustomClassInputNet(torch.nn.Module):
    def forward(self, x, custom_class_obj):
        if custom_class_obj.x == 1:
            return x+1
        return x


class NeuralNetSinglePositionalArgument(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetSinglePositionalArgument, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NeuralNetCustomClassOutput(torch.nn.Module):
    class CustomClass(object):
        def __init__(self, out1, out2, out3):
            self.out1 = out1
            self.out2 = out2
            self.out3 = out3

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetCustomClassOutput, self).__init__()

        self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc1_2 = torch.nn.Linear(hidden_size, num_classes)

        self.fc2_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.fc2_2 = torch.nn.Linear(hidden_size, num_classes)

        self.fc3_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.fc3_2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1, input2, input3):
        out1 = self.fc1_2(self.relu1(self.fc1_1(input1)))
        out2 = self.fc2_2(self.relu2(self.fc2_1(input2)))
        out3 = self.fc3_2(self.relu3(self.fc3_1(input3)))
        return NeuralNetCustomClassOutput.CustomClass(out1, out2, out3)


class MyCustomFunctionReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class MyReLU(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input
        self.relu = MyReLU.apply

    def forward(self, input):
        return self.relu(input)
