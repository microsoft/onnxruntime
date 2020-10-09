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
print('Training MNIST on ORTModule....')
model = ORTModule(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Data loader
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=batch_size,
                                           shuffle=True)
# TODO: Get probability_grad from PyTorch Loss
probability_grad = torch.tensor([
[0.36297542, 0.2297899, -0.10638658, 0.21579745, -0.12323117, -0.35163468, -0.16475351, -0.27790004, 0.20993066, 0.068910174],
[0.30177414, 0.4719398, -0.2290834, 0.61155605, -0.10533161, -0.068530589, -0.16963659, -0.034698304, 0.20859459, 0.071662053],
[0.26006302, 0.59704441, 0.2594507, 0.027483933, 0.17754407, -0.076404758, -0.15315992, -0.3511225, 0.096852496, -0.040248722],
[0.020109242, 0.47963268, 0.16444968, 0.28207836, 0.091335267, -0.34438723, -0.32664698, -0.04607122, 0.16735722, 0.28467956],
[-0.0067059044, 0.49364114, -0.023130134, 0.2933957, -0.12842584, -0.37883937, 0.083117418, -0.28517962, -0.021336049, -0.0058415309],
[-0.075187646, 0.24679491, 0.031593084, 0.59585023, -0.208859, -0.18786775, 0.18447922, -0.074010387, -0.056447648, -0.078843385],
[0.43958831, 0.53015679, -0.16698451, 0.3980948, 0.16000611, -0.016911259, -0.13209809, -0.10536471, 0.00073796883, 0.22187582],
[0.19641832, 0.47633961, 0.14354521, 0.49611267, -0.25266212, -0.28930596, -0.098222524, -0.17880601, 0.3030878, -0.086537011],
[0.16706356, 0.25445995, -0.36106035, 0.3932263, 0.020241318, -0.046459652, -0.30798167, 0.033364233, 0.10860923, 0.161856],
[0.076634176, 0.21363905, 0.14411786, 0.42425469, -0.36067143, -0.024277387, -0.23279551, -0.027842108, 0.11602029, 0.045313828],
[-0.067607164, 0.29514131, -0.21749593, 0.34894356, 0.10760085, -0.10467422, -0.39584625, 0.14010972, 0.21694142, 0.17883658],
[0.11919088, 0.17774329, -0.063672006, 0.31304225, 0.022851272, 0.00603014, -0.063586265, -0.11567068, 0.18024546, -0.044242512],
[0.28452805, 0.28950649, -0.030564137, 0.062676579, 0.037082255, -0.34579667, -0.18721311, -0.048553426, -0.047528304, -0.067283757],
[0.16541988, 0.6750235, 0.36633614, 0.12827933, -0.1848262, -0.12122689, 0.24612407, -0.22443134, 0.29384404, 0.029458519],
[0.022512322, -0.020067703, -0.035412017, 0.042415313, 0.01781881, -0.19647799, -0.019232273, -0.27665097, -0.085087284, -0.23508132],
[-0.056501552, 0.23281966, 0.012086541, 0.34509954, 0.096981436, -0.14569771, -0.24759589, 0.0071231984, 0.32205793, 0.027363759],
[-0.10276053, -0.15549006, 0.026301131, 0.067043148, -0.12606248, 0.042133313, -0.23401891, -0.16697425, -0.03425476, 0.14876992],
[0.20445672, 0.25619513, 0.16442557, 0.077375375, 0.13566223, -0.099527359, -0.12576742, -0.45158958, 0.32187107, 0.092045955],
[0.34017974, -0.066395164, 0.20674077, 0.16103405, -0.27109221, -0.24286765, -0.14018115, -0.0068955906, 0.17458764, -0.072009444],
[-0.081807368, 0.30574301, -0.15613964, 0.33026001, -0.12889105, -0.053762466, 0.036609523, -0.16667747, 0.12113887, -0.10802352],
])

# Training Loop
loss = float('inf')
for iteration, (data, target) in enumerate(train_loader):
    if iteration == 1:
        print(f'Final loss is {loss}')
        break

    data = data.reshape(data.shape[0], -1)
    optimizer.zero_grad()
    probability, intermediates = model(data)
    print(f'Output from forward has shape {probability.size()}')
    loss = criterion(probability, target)

    # Fake backward call to test backprop graph
    fc1_bias_grad, fc1_weight_grad, fc2_weight_grad, fc2_bias_grad = model._run_backward_graph(probability_grad, intermediates, data)
    fc1_bias_grad = torch.nn.Parameter(torch.from_numpy(fc1_bias_grad))
    fc2_bias_grad = torch.nn.Parameter(torch.from_numpy(fc2_bias_grad))
    fc1_weight_grad = torch.nn.Parameter(torch.from_numpy(fc1_weight_grad))
    fc2_weight_grad = torch.nn.Parameter(torch.from_numpy(fc2_weight_grad))
    model._original_module.fc1.bias = fc1_bias_grad
    model._original_module.fc1.weight = fc1_weight_grad
    model._original_module.fc2.bias = fc2_bias_grad
    model._original_module.fc2.weight = fc2_weight_grad

    print(f'Output from backaward has the following shapes after update:')
    print(f'fc1_bias_grad={fc1_bias_grad.size()}')
    print(f'fc2_bias_grad={fc2_bias_grad.size()}')
    print(f'fc1_weight_grad={fc1_weight_grad.size()}')
    print(f'fc2_weight_grad={fc2_weight_grad.size()}')
    # loss.backward(target)
    # optimizer.step()

    if iteration == 0:
        print(f'Initial loss is {loss}')
print('Tah dah!')