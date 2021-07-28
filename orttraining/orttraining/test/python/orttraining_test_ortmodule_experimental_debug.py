
import os
import torch
from onnxruntime.training.ortmodule import ORTModule
from onnxruntime.capi import _pybind_state as C
import onnxruntime.training.ortmodule._experimental.debug as debug_config


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

def test_load_config_from_json():
    path_to_json = os.path.join(os.getcwd(), 'orttraining_test_ortmodule_experimental_debug_config.json')

    device = 'cuda'
    model = ORTModule(Net().to(device))

    debug_config.load_from_json(model, path_to_json)

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

        # test allow layer norm mod precision
        assert ort_model_attributes._allow_layer_norm_mod_precision == True

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
        assert ort_model_attributes._debug_options.save_onnx_models.name_prefix == 'my_model'
        assert ort_model_attributes._debug_options.logging.log_level.name == "VERBOSE"
