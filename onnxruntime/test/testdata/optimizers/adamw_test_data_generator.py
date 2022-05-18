# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used to generate test data for Adam optimizer tests in
# orttraining/orttraining/test/training_ops/cuda/optimizer/adamw_test.cc.

import torch


class SingleParameterModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, input1):
        out = self.fc1(input1)
        return out

class MultipleParametersModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def generate_adamw_test_data(seed, _model_setup_func, data_func, train_step_count, adam_mode):
    def _run_step(model, x, target):
        prediction = model(x)
        criterion = torch.nn.MSELoss()
        loss = criterion(prediction, target)
        loss.backward()

    def _torch_tensor_to_str(t):
        return str(t.detach().cpu().numpy().round(10).astype(str).tolist()).replace("'", "")

    def _str_list_to_str(val_list):
        ss = ""
        for val in val_list:
            ss += val.replace("[", "{").replace("]", "},")
        return ss

    def _to_adam_test_format(data, name_of_the_data):
        print(f"{name_of_the_data} ==================")
        for name, val in data.items():
            print("{{ \"{}\", {{ {} }}, }},".format(name, _str_list_to_str(val)))


    torch.manual_seed(seed)

    # Prepare model, zero gradient.
    pt_model = _model_setup_func()
    pt_model.zero_grad()

    # Prepare optimizer.
    adamw_optimizer = None
    if adam_mode == 0:
        adamw_optimizer = torch.optim.AdamW(pt_model.parameters(), lr=1e-3)
    elif adam_mode == 1:
        from transformers import AdamW
        adamw_optimizer = AdamW(pt_model.parameters(), lr=1e-3)
    else:
        raise ValueError(f"invalid adam_model: {adam_mode}")

    # Build index to name mapping, which is used to retrieve data from optimizer group.
    param_index_to_name_mapping = {}
    index = 0
    for param in pt_model.named_parameters():
        param_index_to_name_mapping[index] = param[0]
        index += 1

    # Dump the optimizer configs, our adam tests should align with these.
    for group in adamw_optimizer.param_groups:
        print(f"beta1, beta2 : {group['betas']}, lr={group['lr']},"
               " weight_decay={group['weight_decay']}, eps={group['eps']})")

    p_dict = {}
    g_dict = {}
    m1_dict = {}
    m2_dict = {}
    for step in range(train_step_count):
        x1, target = data_func()
        _run_step(pt_model, x1, target)
        torch.cuda.synchronize()

        for name, param in pt_model.named_parameters():
            if name not in p_dict:
                p_dict[name] = []
                g_dict[name] = []
                m1_dict[name] = []
                m2_dict[name] = []

            # Collect flattened parameter data.
            p_dict[name].append(_torch_tensor_to_str(param.view(-1)))

            if step != train_step_count -1:
                # Collect flattened gradient data.
                # Skip collecting the last step's gradients.
                g_dict[name].append(_torch_tensor_to_str(param.grad.view(-1)))

        torch.cuda.synchronize()

        for group in adamw_optimizer.param_groups:
            p_index = 0
            for p in group['params']:
                state = adamw_optimizer.state[p]
                name = param_index_to_name_mapping[p_index]
                # Collect flattened optimizer state data.
                if len(state) == 0:
                    m1_dict[name].append(_torch_tensor_to_str(torch.zeros_like(p).view(-1)))
                    m2_dict[name].append(_torch_tensor_to_str(torch.zeros_like(p).view(-1)))
                else:
                    m1_dict[name].append(_torch_tensor_to_str(state['exp_avg'].view(-1)))
                    m2_dict[name].append(_torch_tensor_to_str(state['exp_avg_sq'].view(-1)))
                p_index += 1

        adamw_optimizer.step()
        adamw_optimizer.zero_grad()

    torch.cuda.synchronize()
    _to_adam_test_format(p_dict, "Parameters")
    _to_adam_test_format(g_dict, "Gradients")
    _to_adam_test_format(m1_dict, "Momentum1s")
    _to_adam_test_format(m2_dict, "Momentum2s")


def generate_torch_adamw_single_weight_tests(adam_mode, run_step_count):
    seed=8888
    device = "cuda"
    batch_size, dimension_in, dimension_hidden = 2, 2, 3

    def _model_setup_func():
        pt_model = SingleParameterModule(dimension_in, dimension_hidden).to(device)
        return pt_model

    def _data_func():
        x1 = torch.randn(batch_size, dimension_in, device=device, dtype=torch.float32)
        target = torch.randn(batch_size, dimension_hidden, device=device, dtype=torch.float32)
        return x1, target

    generate_adamw_test_data(seed, _model_setup_func, _data_func, run_step_count, adam_mode)

def generate_torch_adamw_multiple_weights_tests(adam_mode, run_step_count):
    seed=6666
    device = "cuda"
    batch_size, dimension_in, dimension_hidden, DIM_OUT = 2, 2, 3, 2

    def _model_setup_func():
        pt_model = MultipleParametersModule(dimension_in, dimension_hidden, DIM_OUT).to(device)
        return pt_model

    def data_func():
        x1 = torch.randn(batch_size, dimension_in, device=device, dtype=torch.float32)
        target = torch.randn(batch_size, DIM_OUT, device=device, dtype=torch.float32)
        return x1, target

    generate_adamw_test_data(seed, _model_setup_func, data_func, run_step_count, adam_mode)

def main():
    test_data_step_count = 11
    for adam_mode in range(0, 2):
        generate_torch_adamw_single_weight_tests(adam_mode, test_data_step_count)
        generate_torch_adamw_multiple_weights_tests(adam_mode, test_data_step_count)


if __name__ == "__main__":
    main()
