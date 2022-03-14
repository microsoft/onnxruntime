# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import onnxruntime_pybind11_state as torch_ort

device = torch_ort.device()

ones = torch.ones(2, 3).to(device)
print(ones.cpu())

twos = ones + ones
print(twos.cpu())

threes = twos + ones
print(threes.cpu())

fours = twos * twos
print(fours.cpu())

fenced_ten = torch.tensor(
  [[-1, -1, -1],
   [-1, 10, -1],
   [-1, -1, -1]],
  device = device, dtype=torch.float)

print(fenced_ten.numel())
print(fenced_ten.size())
print(fenced_ten.cpu())
print(fenced_ten.relu().cpu())

a = torch.ones(3, 3).to(device)
b = torch.ones(3, 3)
c = a + b
d = torch.sin (c)
e = torch.tan (c)
torch.sin_(c)
print ("sin-in-place:")
print(c.cpu())
print ("sin explicit:")
print (d.cpu ())

a = torch.tensor([[10, 10]], dtype=torch.float).to(device)
b = torch.tensor([[3.3, 3.3]]).to(device)
c = torch.fmod(a, b)

print(c.cpu())

a = torch.tensor([[5, 3, -5]], dtype=torch.float).to(device)
b = torch.hardshrink(a, 3) #should be [5, 0, -5]
c = torch.nn.functional.softshrink(a, 3) #should be [2, 0, -2]
print(b.cpu())
print(c.cpu())