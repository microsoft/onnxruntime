import os
import torch
from onnxruntime.training import ortmodule
from onnxruntime.capi import _pybind_state as C
from onnxruntime.training.ortmodule.experimental.json_config import load_from_json


class Net(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(Net, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def test_load_config_from_json_1():
    device = "cuda"
    model = ortmodule.ORTModule(Net().to(device))

    # load from json once.
    path_to_json = os.path.join(os.getcwd(), "orttraining_test_ortmodule_experimental_json_config_2.json")
    load_from_json(model, path_to_json)

    # load from json another time
    path_to_json = os.path.join(os.getcwd(), "orttraining_test_ortmodule_experimental_json_config_1.json")
    load_from_json(model, path_to_json)

    for training_mode in [True, False]:
        ort_model_attributes = model._torch_module._execution_manager(training_mode)

        # test propagate cast ops
        assert ort_model_attributes._propagate_cast_ops_strategy == C.PropagateCastOpsStrategy.FLOOD_FILL
        assert ort_model_attributes._propagate_cast_ops_level == 3
        assert ort_model_attributes._propagate_cast_ops_allow == ["ABC", "DEF"]

        # test use external gpu allocator
        assert ort_model_attributes._use_external_gpu_allocator == False

        # test enable custom autograd function
        assert ort_model_attributes._enable_custom_autograd_function == True

        # test use static shape
        assert ort_model_attributes._use_static_shape == True

        # test run symbolic shape inference
        assert ort_model_attributes._run_symbolic_shape_infer == False

        # test enable grad acc optimization
        assert ort_model_attributes._enable_grad_acc_optimization == True

        # test skip check
        assert ort_model_attributes._skip_check.value == 14

        # test debug options
        assert ort_model_attributes._debug_options.save_onnx_models.save == True
        assert ort_model_attributes._debug_options.save_onnx_models.name_prefix == "my_model"
        assert ort_model_attributes._debug_options.logging.log_level.name == "VERBOSE"

        # test use memory aware gradient builder.
        assert ort_model_attributes._use_memory_efficient_gradient == False

        # test fallback policy
        assert ort_model_attributes._fallback_manager.policy.value == 1

        # assert onnx opset version
        assert ortmodule.ONNX_OPSET_VERSION == 13


def test_load_config_from_json_2():
    device = "cuda"
    model = ortmodule.ORTModule(Net().to(device))

    # load from json once.
    path_to_json = os.path.join(os.getcwd(), "orttraining_test_ortmodule_experimental_json_config_1.json")
    load_from_json(model, path_to_json)

    # load from json another time
    path_to_json = os.path.join(os.getcwd(), "orttraining_test_ortmodule_experimental_json_config_2.json")
    load_from_json(model, path_to_json)

    for training_mode in [True, False]:
        ort_model_attributes = model._torch_module._execution_manager(training_mode)

        # test propagate cast ops
        assert ort_model_attributes._propagate_cast_ops_strategy == C.PropagateCastOpsStrategy.INSERT_AND_REDUCE
        assert ort_model_attributes._propagate_cast_ops_level == 5
        assert ort_model_attributes._propagate_cast_ops_allow == ["XYZ", "PQR"]

        # test use external gpu allocator
        assert ort_model_attributes._use_external_gpu_allocator == True

        # test enable custom autograd function
        assert ort_model_attributes._enable_custom_autograd_function == False

        # test use static shape
        assert ort_model_attributes._use_static_shape == False

        # test run symbolic shape inference
        assert ort_model_attributes._run_symbolic_shape_infer == True

        # test enable grad acc optimization
        assert ort_model_attributes._enable_grad_acc_optimization == False

        # test skip check
        assert ort_model_attributes._skip_check.value == 10

        # test debug options
        assert ort_model_attributes._debug_options.save_onnx_models.save == True
        assert ort_model_attributes._debug_options.save_onnx_models.name_prefix == "my_other_model"
        assert ort_model_attributes._debug_options.logging.log_level.name == "INFO"

        # test use memory aware gradient builder.
        assert ort_model_attributes._use_memory_efficient_gradient == True

        # test fallback policy
        assert ort_model_attributes._fallback_manager.policy.value == 250

        # assert onnx opset version
        assert ortmodule.ONNX_OPSET_VERSION == 12
