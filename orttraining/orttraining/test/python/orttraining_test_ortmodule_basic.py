import argparse
import torch
from torchvision import datasets, transforms
import torchviz

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

def main():
    #Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--pytorch-only', action='store_true', default=False,
                        help='disables ONNX Runtime training')
    args = parser.parse_args()

    # Model architecture
    lr = 1e-4
    batch_size=20
    seed=42

    torch.manual_seed(seed)
    set_seed(seed)


    model = NeuralNet(input_size=784, hidden_size=500, num_classes=10)
    print('Training MNIST on ORTModule....')
    if not args.pytorch_only:
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
    loss = float('inf')
    for iteration, (data, target) in enumerate(train_loader):
        if iteration == 1:
            print(f'Final loss is {loss}')
            break

        data = data.reshape(data.shape[0], -1)
        optimizer.zero_grad()
        if args.pytorch_only:
            print("Using PyTorch-only API")
            probability = model(data)

            pytorch_backward_graph = torchviz.make_dot(probability, params=dict(list(model.named_parameters())))
            print(f'probability.grad_fn={probability.grad_fn}')
            print(f'probability.grad_fn.next_functions={probability.grad_fn.next_functions}')
            # pytorch_backward_graph.view()
            probability.retain_grad()
        else:
            print("Using ONNX Runtime Flexible API")
            probability, intermediates = model(data)
            probability.requires_grad_(True)

        print(f'Output from forward has shape {probability.size()}')
        loss = criterion(probability, target)
        loss.backward()
        print(f'***** probability.grad[0]={probability.grad[0]}')

        if args.pytorch_only:
            print(f'***** (PYTORCH) fc1.bias_grad[0]          BEFORE {model.fc1.bias.data[0].item()}')
            print(f'***** (PYTORCH) fc1.weight_grad[0][0]     BEFORE {model.fc1.weight.data[0][0].item()}')
            print(f'***** (PYTORCH) fc2.bias_grad[0]          BEFORE {model.fc2.bias.data[0].item()}')
            print(f'***** (PYTORCH) fc2.weight_grad[0][0]     BEFORE {model.fc2.weight.data[0][0].item()}')
        else:
            # import pdb; pdb.set_trace()
            # Fake backward call to test backprop graph
            # TODO: The model output *order* is changing from ONNX export to ONNX export
            fc1_bias_grad, fc1_weight_grad, fc2_weight_grad, fc2_bias_grad = model._run_backward_graph(probability.grad, intermediates, data)
            fc1_bias_grad = torch.from_numpy(fc1_bias_grad).requires_grad_(True)
            fc2_bias_grad = torch.from_numpy(fc2_bias_grad).requires_grad_(True)
            fc1_weight_grad = torch.from_numpy(fc1_weight_grad).requires_grad_(True)
            fc2_weight_grad = torch.from_numpy(fc2_weight_grad).requires_grad_(True)
            fc1_bias_grad.retain_grad()
            fc1_weight_grad.retain_grad()
            fc2_bias_grad.retain_grad()
            fc2_weight_grad.retain_grad()

            print(f'***** (ONNX Runtime) fc1_bias_grad[0]          BEFORE {model._original_module.fc1.bias.data[0].item()}')
            print(f'***** (ONNX Runtime) fc1_weight_grad[0][0]     BEFORE {model._original_module.fc1.weight.data[0][0].item()}')
            print(f'***** (ONNX Runtime) fc2_bias_grad[0]          BEFORE {model._original_module.fc2.bias.data[0].item()}')
            print(f'***** (ONNX Runtime) fc2_weight_grad[0][0]     BEFORE {model._original_module.fc2.weight.data[0][0].item()}')
            print(f'***** (ONNX Runtime) fc1_bias_grad[0]          AFTER {fc1_bias_grad[0].item()}')
            print(f'***** (ONNX Runtime) fc1_weight_grad[0][0]     AFTER {fc1_weight_grad[0][0]}')
            print(f'***** (ONNX Runtime) fc2_bias_grad[0]          AFTER {fc2_bias_grad[0].item()}')
            print(f'***** (ONNX Runtime) fc2_weight_grad[0][0]     AFTER {fc2_weight_grad[0][0].item()}')
            model._original_module.fc1.bias.data = fc1_bias_grad.data
            model._original_module.fc1.weight.data = fc1_weight_grad.data
            model._original_module.fc2.bias.data = fc2_bias_grad.data
            model._original_module.fc2.weight.data = fc2_weight_grad.data

            print(f'Output from backaward has the following shapes after update:')
            print(f'fc1_bias_grad={fc1_bias_grad.size()}')
            print(f'fc2_bias_grad={fc2_bias_grad.size()}')
            print(f'fc1_weight_grad={fc1_weight_grad.size()}')
            print(f'fc2_weight_grad={fc2_weight_grad.size()}')

        optimizer.step()
        if args.pytorch_only:
            print(f'***** (PYTORCH) fc1.bias_grad[0]          AFTER {model.fc1.bias.data[0].item()}')
            print(f'***** (PYTORCH) fc1.weight_grad[0][0]     AFTER {model.fc1.weight.data[0][0].item()}')
            print(f'***** (PYTORCH) fc2.bias_grad[0]          AFTER {model.fc2.bias.data[0].item()}')
            print(f'***** (PYTORCH) fc2.weight_grad[0][0]     AFTER {model.fc2.weight.data[0][0].item()}')

        if iteration == 0:
            print(f'Initial loss is {loss}')
    print('Tah dah!')

if __name__ == '__main__':
    main()
