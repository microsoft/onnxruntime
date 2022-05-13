# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import numpy as np
import torch

# PyTorch model definitions for tests
np.set_printoptions(precision=10)
torch.set_printoptions(precision=10)

class SingleParamterModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleParamterModule, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, input1):
        out = self.fc1(input1)
        return out

class MultipleParamtersModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultipleParamtersModule, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def GenerateAdamWTestData(seed, model_setup_func, data_func, train_step_count):
    torch.manual_seed(seed)
    pt_model = model_setup_func()
    adamw_optimizer = torch.optim.AdamW(pt_model.parameters(), lr=1e-3)

    def run_step(model, x, target):
        prediction = model(x)
        criterion = torch.nn.MSELoss()
        loss = criterion(prediction, target)
        loss.backward()

    def torch_tensor_to_list(t):
        return t.detach().cpu().tolist()

    pt_model.zero_grad()

    p_list = {}
    g_list = {}
    m1_list = {}
    m2_list = {}

    id_to_name = {}
    index = 0
    for ort_param in pt_model.named_parameters():
        id_to_name[index] = ort_param[0]
        index += 1

    for group in adamw_optimizer.param_groups:
        print(f"beta1, beta2 : {group['betas']}, lr={group['lr']}, weight_decay={group['weight_decay']},eps={group['eps']})")

    for step in range(train_step_count):
        x1, target = data_func()
        run_step(pt_model, x1, target)

        torch.cuda.synchronize()
        for name, pt_param in pt_model.named_parameters():
            if name not in p_list:
                p_list[name] = []
                g_list[name] = []
                m1_list[name] = []
                m2_list[name] = []

            p_list[name].append(torch_tensor_to_list(pt_param.view(-1)))
            if step != train_step_count -1:
                # skip collecting the last step's gradients.
                g_list[name].append(torch_tensor_to_list(pt_param.grad.view(-1)))

        torch.cuda.synchronize()

        for group in adamw_optimizer.param_groups:
            p_index = 0
            for p in group['params']:
                state = adamw_optimizer.state[p]
                name = id_to_name[p_index]
                if len(state) == 0:
                    m1_list[name].append(torch_tensor_to_list(torch.zeros_like(p).view(-1)))
                    m2_list[name].append(torch_tensor_to_list(torch.zeros_like(p).view(-1)))
                else:
                    m1_list[name].append(torch_tensor_to_list(state['exp_avg'].view(-1)))
                    m2_list[name].append(torch_tensor_to_list(state['exp_avg_sq'].view(-1)))
                p_index += 1

        adamw_optimizer.step()
        adamw_optimizer.zero_grad()
        torch.cuda.synchronize()

    def val_to_str(val):
        return str(val).replace("[", "{").replace("]", "}")

    print("Parameters:==================")
    for name, val in p_list.items():
        print("name: ", name, "value: ", val_to_str(val))

    print("Gradients:==================")
    for name, val in g_list.items():
        print("name: ", name, "value: ", val_to_str(val))

    print("Momentums1:==================")
    for name, val in m1_list.items():
        print("name: ", name, "value: ", val_to_str(val))

    print("Momentums2:==================")
    for name, val in m2_list.items():
        print("name: ", name, "value: ", val_to_str(val))


def TorchAdamSingleWeightTests():
    def gen(run_step_count):
        seed=8888
        device = "cuda"
        BATCH, DIM_IN, DIM_HIDDEN = 2, 2, 3

        def model_setup_func():
            pt_model = SingleParamterModule(DIM_IN, DIM_HIDDEN).to(device)
            return pt_model

        def data_func():
            x1 = torch.randn(BATCH, DIM_IN, device=device, dtype=torch.float32)
            target = torch.randn(BATCH, DIM_HIDDEN, device=device, dtype=torch.float32)
            return x1, target

        GenerateAdamWTestData(seed, model_setup_func, data_func, run_step_count)

    # Generate data for TorchAdamSingleWeightTest_Loop10Steps.
    gen(11)

def TorchAdamMultipleWeightTests():
    def gen(run_step_count):
        seed=6666
        device = "cuda"
        BATCH, DIM_IN, DIM_HIDDEN, DIM_OUT = 2, 2, 3, 2

        def model_setup_func():
            pt_model = MultipleParamtersModule(DIM_IN, DIM_HIDDEN, DIM_OUT).to(device)
            return pt_model

        def data_func():
            x1 = torch.randn(BATCH, DIM_IN, device=device, dtype=torch.float32)
            target = torch.randn(BATCH, DIM_OUT, device=device, dtype=torch.float32)
            return x1, target

        GenerateAdamWTestData(seed, model_setup_func, data_func, run_step_count)

    # Generate data for TorchAdamMultipleWeightsTest_Loop10Steps.
    gen(11)

TorchAdamMultipleWeightTests()
