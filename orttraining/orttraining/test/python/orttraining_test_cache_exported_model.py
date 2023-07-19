import os
import torch
from onnxruntime.training.ortmodule import ORTModule

os.environ["ORTMODULE_CACHE_DIR"] = "cache_dir"

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.nn.functional.relu(self.fc(x))
        return x

model = Net()
model = ORTModule(model)

data = torch.randn(1, 10)
output = model(data)

assert len(os.listdir("cache_dir")) == 1
