import torch
from torchvision import datasets, transforms

from onnxruntime import set_seed
from onnxruntime.training import ORTModule

import _test_commons
import _test_helpers


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Model architecture
lr = 1e-4
batch_size=20
seed=42

torch.manual_seed(seed)
set_seed(seed)


model = NeuralNet(input_size=784, hidden_size=500, num_classes=10)
model = ORTModule(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Data loader
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=batch_size,
                                           shuffle=True)

# Training Loop
print('Training MNIST on ORTModule....')
loss = float('inf')
for iteration, (data, target) in enumerate(train_loader):
    if iteration == 1:
        print(f'Final loss is {loss}')
        break

    data = data.reshape(data.shape[0], -1)
    optimizer.zero_grad()
    probability = model(data)
    print(f'Output from forward has shape {probability.size()}: {probability}')
    # import pdb; pdb.set_trace()
    loss = criterion(probability, target)
    # loss.backward()
    # optimizer.step()

    if iteration == 0:
        print(f'Initial loss is {loss}')
print('Tah dah!')