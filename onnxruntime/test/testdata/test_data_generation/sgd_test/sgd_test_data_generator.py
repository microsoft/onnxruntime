# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This file is used to generate test data for SGD optimizer tests in
   orttraining/orttraining/test/training_ops/cuda/optimizer/sgd_test.cc."""

import torch


class SingleParameterModule(torch.nn.Module):
    """A dummy module containing only one trainable parameter."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, input1):
        """Module forward call."""
        out = self.fc1(input1)
        return out


class MultipleParametersModule(torch.nn.Module):
    """A dummy module containing multiple trainable parameters."""

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        """Module forward call."""
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def generate_sgd_test_data(seed, _model_setup_func, data_func, train_step_count, json_file_name, device):
    """Generate test data using specified model/data and other configs."""

    is_cuda = device == "cuda"

    def _sync_stream():
        if is_cuda is True:
            torch.cuda.synchronize()

    def _run_step(model, input, target):
        prediction = model(input)
        criterion = torch.nn.MSELoss()
        loss = criterion(prediction, target)
        loss.backward()

    def _torch_tensor_to_str(torch_tensor):
        """Torch tensor to string."""
        return torch_tensor.detach().cpu().numpy().tolist()

    def _build_param_index_to_name_mapping(model, map_result):
        """Build index to name mapping, which is used to retrieve data from optimizer group."""
        for index, param in enumerate(model.named_parameters()):
            map_result[index] = param[0]

    torch.manual_seed(seed)

    # Prepare model, zero gradient.
    pt_model = _model_setup_func()
    pt_model.zero_grad()

    # Prepare optimizer.
    sgd_optimizer = torch.optim.SGD(pt_model.parameters(), lr=1e-3)

    param_index_to_name_mapping = {}
    _build_param_index_to_name_mapping(pt_model, param_index_to_name_mapping)

    # Dump the optimizer configs, our adam tests should align with these.
    for group in sgd_optimizer.param_groups:
        print(f"lr={group['lr']}")

    p_dict = {}
    g_dict = {}
    for step in range(train_step_count):
        input, target = data_func()
        _run_step(pt_model, input, target)
        _sync_stream()

        for name, param in pt_model.named_parameters():
            if name not in p_dict:
                p_dict[name] = []
                g_dict[name] = []

            # Collect flattened parameter data.
            p_dict[name].append(_torch_tensor_to_str(param.view(-1)))
            if step == 0:
                print(f"Weight name: {name}, shape: {param.shape}, data type: {param.dtype}")

            if step != train_step_count - 1:
                # Collect flattened gradient data.
                # Skip collecting the last step's gradients.
                g_dict[name].append(_torch_tensor_to_str(param.grad.view(-1)))

        _sync_stream()

        sgd_optimizer.step()
        sgd_optimizer.zero_grad()

    _sync_stream()
    data = {
        "Parameters": p_dict,
        "Gradients": g_dict,
    }
    import json
    import os

    directory = device
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, json_file_name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def generate_sgd_single_weight_tests(run_step_count, device):
    """Generate test data using specified mode of sgd."""
    seed = 8888
    batch_size, dimension_in, dimension_hidden = 2, 2, 3

    def _model_setup_func():
        pt_model = SingleParameterModule(dimension_in, dimension_hidden).to(device)
        return pt_model

    def _data_func():
        input = torch.randn(batch_size, dimension_in, device=device, dtype=torch.float32)
        target = torch.randn(batch_size, dimension_hidden, device=device, dtype=torch.float32)
        return input, target

    json_file_name = "sgd_test_single_weight.json"
    generate_sgd_test_data(seed, _model_setup_func, _data_func, run_step_count, json_file_name, device)


def generate_sgd_multiple_weights_tests(run_step_count, device):
    """Generate test data using specified mode of sgd."""
    seed = 6666
    batch_size, dimension_in, dimension_hidden, dim_out = 2, 2, 3, 2

    def _model_setup_func():
        pt_model = MultipleParametersModule(dimension_in, dimension_hidden, dim_out).to(device)
        return pt_model

    def data_func():
        input = torch.randn(batch_size, dimension_in, device=device, dtype=torch.float32)
        target = torch.randn(batch_size, dim_out, device=device, dtype=torch.float32)
        return input, target

    json_file_name = "sgd_test_multiple_weights.json"
    generate_sgd_test_data(seed, _model_setup_func, data_func, run_step_count, json_file_name, device)


def main():
    """Main entry."""
    device_candidates = ["cuda", "cpu"]
    test_data_step_count = 11
    for device in device_candidates:
        generate_sgd_single_weight_tests(test_data_step_count, device)
        generate_sgd_multiple_weights_tests(test_data_step_count, device)


if __name__ == "__main__":
    main()
