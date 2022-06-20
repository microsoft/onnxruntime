from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import onnxruntime_pybind11_state as torch_ort


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = 784
hidden_size = 500
num_classes = 10
batch_size = 128

batch = torch.rand((batch_size, input_size))
device = torch_ort.device()

with torch.no_grad():

    model = NeuralNet(input_size, hidden_size, num_classes)
    pred = model(batch)
    print("inference result is: ")
    print(pred)

    model.to(device)

    ort_batch = batch.to(device)
    ort_pred = model(ort_batch)
    print("ORT inference result is:")
    print(ort_pred.cpu())
    print("Compare result:")
    print(torch.allclose(pred, ort_pred.cpu(), atol=1e-6))
