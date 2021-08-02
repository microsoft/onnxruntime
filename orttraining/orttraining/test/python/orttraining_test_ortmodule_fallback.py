# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import copy
import torch
import pytest
import warnings

from onnxruntime.training.ortmodule import ORTModule, _utils, _io, DebugOptions, LogLevel, _fallback, TORCH_CPP_DIR
from onnxruntime.training.ortmodule.torch_cpp_extensions import is_installed as is_torch_cpp_extensions_installed
import _test_helpers
from _orttraining_ortmodule_models import NeuralNetSinglePositionalArgument, NeuralNetCustomClassOutput, MyStrNet

# PyTorch model definitions for tests

@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
@pytest.mark.parametrize("matching_policy", [True, False])
def test_ortmodule_fallback_forward(is_training, fallback_enabled, matching_policy):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True results in PyTorch executing the forward graph instead of ORT backend
    # matching_policy: True results in properly matching FALLBACK_UNSUPPORTED_INPUT policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_INPUT
        else:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
    else:
        policy = _fallback._FallbackPolicy.FALLBACK_DISABLE

    from dataclasses import dataclass
    @dataclass
    class Point:
        x: int
        y: int

    class UnsupportedInputModel(torch.nn.Module):
        def __init__(self):
            super(UnsupportedInputModel, self).__init__()
        def forward(self, point):
            return point.x * point.y

    pt_model = UnsupportedInputModel()
    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(fallback_policy=policy))
    inputs = Point(x=2,y=3)

    ort_model.train(is_training)
    pt_model.train(is_training)

    for _ in range(3):
        if fallback_enabled:
            if matching_policy:
                ort_out = ort_model(inputs)
                pt_out = pt_model(inputs)
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                    ort_model(inputs)
                assert "ORTModule does not support the following model data type" in str(type_error.value)
        else:
            with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                ort_model(inputs)
            assert "ORTModule does not support the following model data type" in str(type_error.value)


@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
@pytest.mark.parametrize("matching_policy", [True, False])
def test_ortmodule_fallback_device__multiple(is_training, fallback_enabled, matching_policy):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True results in PyTorch executing the forward graph instead of ORT backend
    # matching_policy: True results in properly matching FALLBACK_UNSUPPORTED_DEVICE policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_INPUT) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
        else:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_INPUT
    else:
        policy = _fallback._FallbackPolicy.FALLBACK_DISABLE

    class ManyDevicesNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
            self.relu = torch.nn.ReLU()
            self.net2 = torch.nn.Linear(10, 5).to('cpu')

        def forward(self, x):
            x = self.relu(self.net1(x.to('cuda:0')))
            return self.net2(x.to('cpu'))

    pt_model = ManyDevicesNet()
    inputs = torch.randn(20, 10)

    for _ in range(3):
        if fallback_enabled:
            if matching_policy:
                ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(fallback_policy=policy))
                pt_model.train(is_training)
                ort_model.train(is_training)
                ort_out = ort_model(inputs)
                pt_out = pt_model(inputs)
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                    # Initialize with fallback policy because Exception will happen during __init__
                    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(fallback_policy=policy))
                assert "ORTModule supports a single device per model" in str(type_error.value)
        else:
            with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                # Initialize with fallback policy because Exception will happen during __init__
                ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(fallback_policy=policy))
            assert "ORTModule supports a single device per model" in str(type_error.value)


@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
@pytest.mark.parametrize("matching_policy", [True, False])
def test_ortmodule_fallback_device__mismatch(is_training, fallback_enabled, matching_policy):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True results in PyTorch executing the forward graph instead of ORT backend
    # matching_policy: True results in properly matching FALLBACK_UNSUPPORTED_DEVICE policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_INPUT) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
        else:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_INPUT
    else:
        policy = _fallback._FallbackPolicy.FALLBACK_DISABLE

    data_device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(fallback_policy=policy))
    pt_model.train(is_training)
    ort_model.train(is_training)
    # For initial model export, use same device for data and model so that PyTorch model can be traced during export
    _ = ort_model(torch.randn(N, D_in))

    # Use data in different device for testing
    inputs = torch.randn(N, D_in, device=data_device)
    ort_model_device = ort_model._torch_module._execution_manager(ort_model._is_training())._device
    input_device = torch.device(inputs.device)

    for _ in range(3):
        if fallback_enabled:
            if matching_policy:
                with pytest.raises(RuntimeError) as e:
                    ort_model(inputs)
                assert "Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!" in str(e.value)
            else:
                with pytest.raises(_fallback.ORTModuleDeviceException) as e:
                    ort_model(inputs)
                assert f"Input argument to forward found on device {input_device}, but expected it to be on module device {ort_model_device}." in str(e.value)
        else:
            with pytest.raises(_fallback.ORTModuleDeviceException) as e:
                ort_model(inputs)
            assert f"Input argument to forward found on device {input_device}, but expected it to be on module device {ort_model_device}." in str(e.value)

@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
@pytest.mark.parametrize("matching_policy", [True, False])
def test_ortmodule_fallback_output(is_training, fallback_enabled, matching_policy):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True results in PyTorch executing the forward graph instead of ORT backend
    # matching_policy: True results in properly matching FALLBACK_UNSUPPORTED_OUTPUT policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_OUTPUT
        else:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
    else:
        policy = _fallback._FallbackPolicy.FALLBACK_DISABLE

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetCustomClassOutput(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(fallback_policy=policy))
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    ort_model.train(is_training)
    pt_model.train(is_training)

    for _ in range(3):
        if fallback_enabled:
            if matching_policy:
                ort_out = ort_model(x, y, z)
                pt_out = pt_model(x, y, z)
                _test_helpers.assert_values_are_close(ort_out.out1, pt_out.out1, rtol=0, atol=0)
                _test_helpers.assert_values_are_close(ort_out.out2, pt_out.out2, rtol=0, atol=0)
                _test_helpers.assert_values_are_close(ort_out.out3, pt_out.out3, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleIOError) as runtime_error:
                    ort_model(x, y, z)
                assert 'ORTModule does not support the following model output type' in str(runtime_error.value)
        else:
            with pytest.raises(_fallback.ORTModuleIOError) as runtime_error:
                ort_model(x, y, z)
            assert 'ORTModule does not support the following model output type' in str(runtime_error.value)

@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
@pytest.mark.parametrize("matching_policy", [True, False])
def test_ortmodule_fallback_input(is_training, fallback_enabled, matching_policy):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True results in PyTorch executing the forward graph instead of ORT backend
    # matching_policy: True results in properly matching FALLBACK_UNSUPPORTED_INPUT policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_INPUT
        else:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
    else:
        policy = _fallback._FallbackPolicy.FALLBACK_DISABLE

    pt_model = MyStrNet()
    ort_model = ORTModule(copy.deepcopy(pt_model), DebugOptions(fallback_policy=policy))
    inputs = torch.randn(1, 2)

    ort_model.train(is_training)
    pt_model.train(is_training)

    for _ in range(3):
        if fallback_enabled:
            if matching_policy:
                ort_out = ort_model(inputs, 'hello')
                pt_out = pt_model(inputs, 'hello')
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleIOError) as ex_info:
                    _ = ort_model(torch.randn(1, 2), 'hello')
                assert "ORTModule does not support the following model data type <class 'str'>" in str(ex_info.value)
        else:
            with pytest.raises(_fallback.ORTModuleIOError) as ex_info:
                _ = ort_model(torch.randn(1, 2), 'hello')
            assert "ORTModule does not support the following model data type <class 'str'>" in str(ex_info.value)


@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
@pytest.mark.parametrize("matching_policy", [True, False])
def test_ortmodule_fallback_torch_model(is_training, fallback_enabled, matching_policy):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True results in PyTorch executing the forward graph instead of ORT backend
    # matching_policy: True results in properly matching FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_TORCH_MODEL
        else:
            policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
    else:
        policy = _fallback._FallbackPolicy.FALLBACK_DISABLE

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    x = torch.randn(N, D_in, device=device)

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model = torch.nn.DataParallel(pt_model)

    for _ in range(3):
        if fallback_enabled:
            if matching_policy:
                ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
                ort_model.train(is_training)
                pt_model.train(is_training)

                ort_out = ort_model(x)
                pt_out = pt_model(x)
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleTorchModelException) as ex_info:
                    ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
                assert "ORTModule is not compatible with torch.nn.DataParallel" in str(ex_info.value)
        else:
            with pytest.raises(_fallback.ORTModuleTorchModelException) as ex_info:
                # Initialize with fallback policy because Exception will happen during __init__
                ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
            assert "ORTModule is not compatible with torch.nn.DataParallel" in str(ex_info.value)




@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
@pytest.mark.parametrize("matching_policy", [True, False])
def test_ortmodule_fallback_init__torch_version(is_training, fallback_enabled, matching_policy):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True results in PyTorch executing the forward graph instead of ORT backend
    # matching_policy: True results in properly matching FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    from packaging import version
    from onnxruntime.training.ortmodule import MINIMUM_RUNTIME_PYTORCH_VERSION_STR
    runtime_pytorch_version = version.parse(torch.__version__.split('+')[0])
    minimum_runtime_pytorch_version = version.parse(MINIMUM_RUNTIME_PYTORCH_VERSION_STR)
    if runtime_pytorch_version < minimum_runtime_pytorch_version:

        if fallback_enabled:
            if matching_policy:
                policy = _fallback._FallbackPolicy.FALLBACK_BAD_INITIALIZATION
            else:
                policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
        else:
            policy = _fallback._FallbackPolicy.FALLBACK_DISABLE

        device = 'cuda'
        N, D_in, H, D_out = 64, 784, 500, 10
        x = torch.randn(N, D_in, device=device)

        pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

        for _ in range(3):
            if fallback_enabled:
                if matching_policy:
                    ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
                    ort_model.train(is_training)
                    pt_model.train(is_training)

                    ort_out = ort_model(x)
                    pt_out = pt_model(x)
                    _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
                else:
                    with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                        ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
                    assert "ONNX Runtime ORTModule frontend requires PyTorch version greater or equal to" in str(ex_info.value)
            else:
                with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                    # Initialize with fallback policy because Exception will happen during __init__
                    ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
                assert "ONNX Runtime ORTModule frontend requires PyTorch version greater or equal to" in str(ex_info.value)
    else:
        warnings.warn('Skipping test_ortmodule_fallback_torch_version.'
                      f' It requires PyTorch prior to {MINIMUM_RUNTIME_PYTORCH_VERSION_STR}')


@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
@pytest.mark.parametrize("matching_policy", [True, False])
def test_ortmodule_fallback_init__missing_cpp_extensions(is_training, fallback_enabled, matching_policy):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True results in PyTorch executing the forward graph instead of ORT backend
    # matching_policy: True results in properly matching FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if is_torch_cpp_extensions_installed(TORCH_CPP_DIR):
        warnings.warn('Skipping test_ortmodule_fallback_init__missing_cpp_extensions.'
                      f' It requires PyTorch CPP extensions to be missing')
    else:
        if fallback_enabled:
            if matching_policy:
                policy = _fallback._FallbackPolicy.FALLBACK_BAD_INITIALIZATION
            else:
                policy = _fallback._FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE
        else:
            policy = _fallback._FallbackPolicy.FALLBACK_DISABLE

        device = 'cuda'
        N, D_in, H, D_out = 64, 784, 500, 10
        x = torch.randn(N, D_in, device=device)

        pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

        for _ in range(3):
            if fallback_enabled:
                if matching_policy:
                    ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
                    ort_model.train(is_training)
                    pt_model.train(is_training)

                    ort_out = ort_model(x)
                    pt_out = pt_model(x)
                    _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
                else:
                    with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                        ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
                    assert "ORTModule's extensions were not detected" in str(ex_info.value)
            else:
                with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                    # Initialize with fallback policy because Exception will happen during __init__
                    ort_model = ORTModule(pt_model, DebugOptions(fallback_policy=policy))
                assert "ORTModule's extensions were not detected" in str(ex_info.value)
