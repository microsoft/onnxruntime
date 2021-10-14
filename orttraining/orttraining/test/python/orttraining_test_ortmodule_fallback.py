# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import copy
import itertools
import os
import torch
import pytest
import warnings

from onnxruntime.training.ortmodule import ORTModule, _fallback, ORTMODULE_TORCH_CPP_DIR
from onnxruntime.training.ortmodule.torch_cpp_extensions import is_installed as is_torch_cpp_extensions_installed
import _test_helpers
from _orttraining_ortmodule_models import (NeuralNetSinglePositionalArgument,
                                           NeuralNetCustomClassOutput,
                                           MyStrNet,
                                           MyCustomFunctionReluModel)

# PyTorch model definitions for tests


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_forward(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_FORCE_TORCH_FORWARD policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = 'FALLBACK_FORCE_TORCH_FORWARD'
        else:
            policy = 'FALLBACK_UNSUPPORTED_DEVICE'
    else:
        policy = 'FALLBACK_DISABLE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

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
    ort_model = ORTModule(copy.deepcopy(pt_model))
    inputs = Point(x=2, y=3)

    ort_model.train(is_training)
    pt_model.train(is_training)

    for i in range(3):
        if fallback_enabled:
            if matching_policy:
                if i > 0 and persist_fallback:
                    assert ort_model._torch_module._execution_manager(
                        is_training=is_training)._fallback_manager._exception is not None
                ort_out = ort_model(inputs)
                pt_out = pt_model(inputs)
                assert ort_out == pt_out
            else:
                with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                    ort_model(inputs)
                assert "ORTModule does not support the following model data type" in str(type_error.value)
        else:
            with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                ort_model(inputs)
            assert "ORTModule does not support the following model data type" in str(type_error.value)


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_device__multiple(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_DEVICE policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DATA) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = 'FALLBACK_UNSUPPORTED_DEVICE'
        else:
            policy = 'FALLBACK_UNSUPPORTED_DATA'
    else:
        policy = 'FALLBACK_DISABLE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

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
                ort_model = ORTModule(copy.deepcopy(pt_model))
                pt_model.train(is_training)
                ort_model.train(is_training)
                ort_out = ort_model(inputs)
                pt_out = pt_model(inputs)
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                    # Initialize with fallback policy because Exception will happen during __init__
                    ort_model = ORTModule(copy.deepcopy(pt_model))
                assert "ORTModule supports a single device per model" in str(type_error.value)
        else:
            with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                # Initialize with fallback policy because Exception will happen during __init__
                ort_model = ORTModule(copy.deepcopy(pt_model))
            assert "ORTModule supports a single device per model" in str(type_error.value)


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_device__mismatch(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_DEVICE policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DATA) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = 'FALLBACK_UNSUPPORTED_DEVICE'
        else:
            policy = 'FALLBACK_UNSUPPORTED_DATA'
    else:
        policy = 'FALLBACK_DISABLE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)
    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    data_device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    ort_model = ORTModule(copy.deepcopy(pt_model))
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
                assert \
                    ("Tensor for argument #1 'self' is on CPU, but expected them to be on GPU (while checking arguments for addmm)" in str(e.value)) \
                    or ("Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!" in str(e.value))
            else:
                with pytest.raises(_fallback.ORTModuleDeviceException) as e:
                    ort_model(inputs)
                assert (f"Input argument to forward found on device {input_device}, "
                        f"but expected it to be on module device {ort_model_device}." in str(e.value))
        else:
            with pytest.raises(_fallback.ORTModuleDeviceException) as e:
                ort_model(inputs)
            assert (f"Input argument to forward found on device {input_device}, "
                    f"but expected it to be on module device {ort_model_device}." in str(e.value))

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']

@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_output(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_DATA policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = 'FALLBACK_UNSUPPORTED_DATA'
        else:
            policy = 'FALLBACK_UNSUPPORTED_DEVICE'
    else:
        policy = 'FALLBACK_DISABLE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    pt_model = NeuralNetCustomClassOutput(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_in, device=device)
    z = torch.randn(N, D_in, device=device)

    ort_model.train(is_training)
    pt_model.train(is_training)

    for i in range(3):
        if fallback_enabled:
            if matching_policy:
                if i > 0 and persist_fallback:
                    assert ort_model._torch_module._execution_manager(
                        is_training=is_training)._fallback_manager._exception is not None
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


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_input(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_DATA policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = 'FALLBACK_UNSUPPORTED_DATA'
        else:
            policy = 'FALLBACK_UNSUPPORTED_DEVICE'
    else:
        policy = 'FALLBACK_DISABLE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

    pt_model = MyStrNet()
    ort_model = ORTModule(copy.deepcopy(pt_model))
    inputs = torch.randn(1, 2)

    ort_model.train(is_training)
    pt_model.train(is_training)

    for i in range(3):
        if fallback_enabled:
            if matching_policy:
                if i > 0 and persist_fallback:
                    assert ort_model._torch_module._execution_manager(
                        is_training=is_training)._fallback_manager._exception is not None
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


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_torch_model(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = 'FALLBACK_UNSUPPORTED_TORCH_MODEL'
        else:
            policy = 'FALLBACK_UNSUPPORTED_DEVICE'
    else:
        policy = 'FALLBACK_DISABLE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

    device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10
    x = torch.randn(N, D_in, device=device)

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)
    pt_model = torch.nn.DataParallel(pt_model)

    for _ in range(3):
        if fallback_enabled:
            if matching_policy:
                ort_model = ORTModule(pt_model)
                ort_model.train(is_training)
                pt_model.train(is_training)

                ort_out = ort_model(x)
                pt_out = pt_model(x)
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleTorchModelException) as ex_info:
                    ort_model = ORTModule(pt_model)
                assert "ORTModule is not compatible with torch.nn.DataParallel" in str(ex_info.value)
        else:
            with pytest.raises(_fallback.ORTModuleTorchModelException) as ex_info:
                # Initialize with fallback policy because Exception will happen during __init__
                ort_model = ORTModule(pt_model)
            assert "ORTModule is not compatible with torch.nn.DataParallel" in str(ex_info.value)


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_init__torch_version(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    from packaging import version
    from onnxruntime.training.ortmodule import MINIMUM_RUNTIME_PYTORCH_VERSION_STR
    runtime_pytorch_version = version.parse(torch.__version__.split('+')[0])
    minimum_runtime_pytorch_version = version.parse(MINIMUM_RUNTIME_PYTORCH_VERSION_STR)
    if runtime_pytorch_version < minimum_runtime_pytorch_version:

        if fallback_enabled:
            if matching_policy:
                policy = 'FALLBACK_BAD_INITIALIZATION'
            else:
                policy = 'FALLBACK_UNSUPPORTED_DEVICE'
        else:
            policy = 'FALLBACK_DISABLE'
        os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
        os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

        device = 'cuda'
        N, D_in, H, D_out = 64, 784, 500, 10
        x = torch.randn(N, D_in, device=device)

        pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

        for i in range(3):
            if fallback_enabled:
                if matching_policy:
                    ort_model = ORTModule(pt_model)
                    ort_model.train(is_training)
                    pt_model.train(is_training)

                    ort_out = ort_model(x)
                    pt_out = pt_model(x)
                    _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
                else:
                    with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                        ort_model = ORTModule(pt_model)
                    assert "ONNX Runtime ORTModule frontend requires PyTorch version greater or equal to" in str(
                        ex_info.value)
            else:
                with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                    # Initialize with fallback policy because Exception will happen during __init__
                    ort_model = ORTModule(pt_model)
                assert "ONNX Runtime ORTModule frontend requires PyTorch version greater or equal to" in str(ex_info.value)
    else:
        warnings.warn('Skipping test_ortmodule_fallback_torch_version.'
                      f' It requires PyTorch prior to {MINIMUM_RUNTIME_PYTORCH_VERSION_STR}')


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_init__missing_cpp_extensions(is_training, fallback_enabled, matching_policy,
                                                         persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if is_torch_cpp_extensions_installed(ORTMODULE_TORCH_CPP_DIR):
        warnings.warn('Skipping test_ortmodule_fallback_init__missing_cpp_extensions.'
                      f' It requires PyTorch CPP extensions to be missing')
    else:

        if fallback_enabled:
            if matching_policy:
                policy = 'FALLBACK_BAD_INITIALIZATION'
            else:
                policy = 'FALLBACK_UNSUPPORTED_DEVICE'
        else:
            policy = 'FALLBACK_DISABLE'
        os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
        os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

        device = 'cuda'
        N, D_in, H, D_out = 64, 784, 500, 10
        x = torch.randn(N, D_in, device=device)

        pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

        for _ in range(3):
            if fallback_enabled:
                if matching_policy:
                    ort_model = ORTModule(pt_model)
                    ort_model.train(is_training)
                    pt_model.train(is_training)

                    ort_out = ort_model(x)
                    pt_out = pt_model(x)
                    _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
                else:
                    with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                        ort_model = ORTModule(pt_model)
                    assert "ORTModule's extensions were not detected" in str(ex_info.value)
            else:
                with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                    # Initialize with fallback policy because Exception will happen during __init__
                    ort_model = ORTModule(pt_model)
                assert "ORTModule's extensions were not detected" in str(ex_info.value)


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_onnx_model__custom_autograd(is_training, fallback_enabled, matching_policy,
                                                        persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_ONNX_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = 'FALLBACK_UNSUPPORTED_ONNX_MODEL'
        else:
            policy = 'FALLBACK_UNSUPPORTED_DEVICE'
    else:
        policy = 'FALLBACK_DISABLE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

    dtype = torch.float
    device = torch.device("cuda")
    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    pt_model = MyCustomFunctionReluModel()
    ort_model = ORTModule(copy.deepcopy(pt_model))
    ort_model.train(is_training)
    pt_model.train(is_training)

    for i in range(3):
        if fallback_enabled:
            if matching_policy:
                if i > 0 and persist_fallback:
                    assert ort_model._torch_module._execution_manager(
                        is_training=is_training)._fallback_manager._exception is not None
                pt_out = pt_model(x.mm(w1)).mm(w2)
                ort_out = ort_model(x.mm(w1)).mm(w2)
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleONNXModelException) as ex_info:
                    _ = ort_model(x.mm(w1)).mm(w2)
                assert "There was an error while exporting the PyTorch model to ONNX" in str(ex_info.value)
        else:
            with pytest.raises(_fallback.ORTModuleONNXModelException) as ex_info:
                # Initialize with fallback policy because Exception will happen during __init__
                _ = ort_model(x.mm(w1)).mm(w2)
            assert "There was an error while exporting the PyTorch model to ONNX" in str(ex_info.value)


@pytest.mark.parametrize("is_training,fallback_enabled,matching_policy,persist_fallback",
                         list(itertools.product([True, False], repeat=4)))
def test_ortmodule_fallback_onnx_model__missing_op(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_ONNX_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = 'FALLBACK_UNSUPPORTED_ONNX_MODEL'
        else:
            policy = 'FALLBACK_UNSUPPORTED_DEVICE'
    else:
        policy = 'FALLBACK_DISABLE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)

    class CrossModule(torch.nn.Module):
        def forward(self, x, y):
            return torch.cross(x, y)
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    pt_model = CrossModule()
    ort_model = ORTModule(copy.deepcopy(pt_model))
    ort_model.train(is_training)
    pt_model.train(is_training)

    for i in range(3):
        if fallback_enabled:
            if matching_policy:
                if i > 0 and persist_fallback:
                    assert ort_model._torch_module._execution_manager(
                        is_training=is_training)._fallback_manager._exception is not None
                pt_out = pt_model(x, y)
                ort_out = ort_model(x, y)
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleONNXModelException) as ex_info:
                    _ = ort_model(x, y)
                assert "There was an error while exporting the PyTorch model to ONNX" in str(ex_info.value)
        else:
            with pytest.raises(_fallback.ORTModuleONNXModelException) as ex_info:
                # Initialize with fallback policy because Exception will happen during __init__
                _ = ort_model(x, y)
            assert "There was an error while exporting the PyTorch model to ONNX" in str(ex_info.value)


@pytest.mark.parametrize("is_training,persist_fallback",
                         list(itertools.product([True, False], repeat=2)))
def test_ortmodule_fallback_warn_message(is_training, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise

    policy = 'FALLBACK_UNSUPPORTED_DEVICE'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy
    os.environ['ORTMODULE_FALLBACK_RETRY'] = str(not persist_fallback)
    os.environ['ORTMODULE_SKIPCHECK_POLICY'] = 'SKIP_CHECK_DISABLED'

    data_device = 'cuda'
    N, D_in, H, D_out = 64, 784, 500, 10

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    pt_model.train(is_training)
    ort_model.train(is_training)
    # For initial model export, use same device for data and model so that PyTorch model can be traced during export
    _ = ort_model(torch.randn(N, D_in))

    # Use data in different device for testing
    inputs = torch.randn(N, D_in, device=data_device)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            with pytest.warns(UserWarning) as warning_record:
                ort_model(inputs)
        assert "Fallback to PyTorch due to exception" in str(warning_record[0].message.args[0])

    del os.environ['ORTMODULE_SKIPCHECK_POLICY']



@pytest.mark.parametrize("is_training,torch_forward",
                         list(itertools.product([True, False], repeat=1)))
def test_ortmodule_fallback_sparse(is_training, torch_forward):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # Validate fix for issue: https://github.com/pytorch/ort/issues/92

    policy = ('FALLBACK_UNSUPPORTED_DEVICE|FALLBACK_UNSUPPORTED_DATA|FALLBACK_UNSUPPORTED_TORCH_MODEL|'
              'FALLBACK_UNSUPPORTED_ONNX_MODEL')
    if torch_forward:
        policy += '|FALLBACK_FORCE_TORCH_FORWARD'
    os.environ['ORTMODULE_FALLBACK_POLICY'] = policy

    class PositionalEncoding(nn.Module):

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            position = torch.arange(max_len).unsqueeze(1)
            div_term = (torch.exp(torch.arange(0, d_model, 2) * 
                        (-math.log(10000.0) / d_model)))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x: Tensor) -> Tensor:
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)   


    class TransformerModel(nn.Module):

        def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                     nlayers: int, dropout: float = 0.5):
            super().__init__()
            self.model_type = 'Transformer'
            encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            self.encoder = nn.Embedding(ntoken, d_model)
            self.d_model = d_model
            self.decoder = nn.Linear(d_model, ntoken)
            self.init_weights()

        def init_weights(self) -> None:
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
            src = self.encoder(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, src_mask)
            output = self.decoder(output)
            return output


    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target


    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = [9, 1352, 0, 9, 1352, 0, 26, 31, 818, 82, 2, 759, 5, 1127, 1376, 3, 23, 30, 8, 2313, 11, 4493, 440, 13, 1, 759, 149, 1, 2433, 6, 533, 3, 37, 10, 672, 22, 8, 4493, 440, 6, 1, 387, 7522, 651, 22, 4472, 8308, 2, 32, 10, 906, 6, 1605, 24, 1, 396, 721, 1127, 3, 23, 30, 8, 2313, 440, 6, 1, 759, 149, 3737, 400, 0, 6, 785, 3, 6, 786, 0, 1556, 8, 440, 17, 3619, 6, 1, 426, 8382, 12, 19, 680, 4, 1, 759, 149, 1, 211, 4881, 23, 1587, 1613, 1377, 676, 397, 5, 5709, 7633, 3, 23, 10, 1143, 6, 1, 697, 1127, 4400, 4, 1, 3345, 8138, 387, 5040, 5861, 2, 32, 10, 906, 24, 1, 3237, 1127, 6, 7952, 5, 1, 0, 0, 4870, 6, 531, 3, 23, 10, 794, 22, 400, 0, 5, 1587, 1613, 2087, 0, 2, 2794, 0, 2, 4927, 3740, 2, 5855, 0, 2, 8261, 8299, 5, 4820, 504, 3, 6, 433, 2, 0, 1587, 1613, 0, 6, 1, 387, 0, 651, 22, 676, 0, 3, 23, 614, 13, 8, 433, 426, 4, 1, 759, 149, 2, 3653, 2, 672, 22, 8, 440, 6, 1, 483, 1127, 317, 4, 529, 7, 4126, 794, 22, 0, 0, 3, 529, 7, 4126, 10, 906, 24, 3571, 1127, 6, 1, 531, 6866, 4, 0, 5, 7437, 3, 0, 1587, 6, 43, 1203, 6, 363, 2, 0, 0, 22, 7383, 2541, 0, 2, 5, 0, 3828, 794, 22, 0, 6852, 3, 6, 86, 363, 2, 0, 88, 8, 2313, 684, 13, 8, 43, 11, 162, 426, 3539, 4, 1, 759, 149, 8509, 1, 585, 2, 672, 22, 31, 684, 13, 1, 759, 149, 0, 6, 200, 363, 3, 23, 30, 8, 4422, 440, 6, 565, 2926, 4, 1, 759, 149, 0, 6, 350, 2, 17, 0, 5842, 3, 0, 1587, 6, 1, 394, 82, 0, 794, 22, 2541, 0, 3, 9, 9, 276, 9, 9, 9, 9, 9, 533, 48, 697, 9, 9, 9, 6, 533, 0, 30, 8, 2313, 11, 4493, 440, 13, 1, 759, 149, 1, 2433, 23, 3814, 2189, 7903, 6, 1, 426, 2, 6, 4454, 1463, 3, 0, 1587, 17, 2189, 6, 1, 387, 7522, 651, 22, 4472, 8308, 2, 32, 10, 906, 6, 1605, 24, 1, 396, 721, 1127, 3, 8, 1289, 4, 0, 12, 19, 827, 6, 1, 1552, 13, 2384, 376, 70, 17, 10338, 7761, 6, 1, 440, 2, 5, 23, 333, 1014, 2031, 6, 1, 7520, 2, 5, 1201, 1586, 3, 23, 614, 6, 1, 759, 149, 3737, 400, 0, 6, 785, 17, 0, 0, 6, 1, 426, 1224, 0, 2, 5, 30, 8, 440, 17, 8, 535, 718, 12129, 11916, 13, 1, 2433, 3, 23, 30, 8, 4422, 440, 6, 521, 13, 43, 2926, 4, 1, 2433, 2, 17, 718, 9435, 3364, 3, 6, 786, 0, 1556, 8, 440, 17, 3619, 6, 1, 426, 8382, 12, 19, 680, 4, 1, 759, 149, 1, 211, 4881, 23, 1587, 1613, 1377, 676, 397, 5, 5709, 7633, 3, 0, 1587, 17, 9574, 2, 6, 1, 697, 1127, 4400, 4, 1, 3345, 8138, 387, 5040, 5861, 3, 29, 10, 906, 24, 1, 3237, 1127, 6, 7952, 2, 5, 1, 0, 0, 4870, 6, 531, 3, 23, 10, 794, 22, 400, 0, 5, 1587, 1613, 2087, 0, 2, 2794, 0, 2, 4927, 3740, 2, 5855, 0, 2, 8261, 8299, 5, 4820, 504, 3, 0, 333, 8, 1846, 1289, 6, 1, 2111, 3907, 1, 2245, 26, 0, 2721, 2, 21, 0, 1888, 28, 2087, 0, 15, 250, 0, 28, 25, 827, 17, 12177, 0, 12, 19, 1639, 16, 2, 1352, 0, 2, 2794, 0, 5, 5855, 0, 3, 1, 4917, 1118, 2, 2087, 0, 5, 1352, 0, 2754, 0, 8897, 1, 0, 3, 9, 9, 9, 433, 48, 677, 9, 9, 9, 6, 433, 0, 1587, 6, 1, 387, 0, 651, 22, 676, 0, 3, 1, 387, 10, 162, 4, 8, 149, 32, 1151, 535, 0, 2, 2050, 6898, 103, 0, 103, 0, 3, 6, 8, 433, 1994, 2, 2131, 1376, 2087, 0, 1550, 0, 17, 44, 4, 25, 2934, 1956, 11, 2203, 120, 7716, 887, 21, 8, 4919, 190, 1352, 0, 2, 58, 10, 6, 1, 760, 2433, 4, 6898, 2, 0, 5, 0, 24, 1, 137, 3, 23, 369, 1028, 1390, 6, 5040, 5861, 3, 23, 3814, 3735, 8458, 13, 1, 433, 426, 4, 1, 759, 149, 2, 3653, 2, 2050, 1359, 120, 0, 3, 0, 1587, 17, 665, 6, 1, 483, 317, 4, 529, 7, 4126, 794, 22, 0, 0, 3, 529, 7, 4126, 10, 906, 24, 3571, 1127, 6, 1, 531, 6866, 4, 0, 5, 7437, 3, 6, 8, 1289, 4, 1, 317, 20, 1, 2111, 3907, 2, 1127, 1724, 928, 11867, 1118, 2, 1352, 0, 4702, 8, 12148, 3113, 7, 1, 1361, 17, 665, 3, 0, 1587, 6, 43, 1203, 6, 363, 2, 0, 0, 22, 7383, 2541, 0, 2, 5, 0, 3828, 794, 22, 0, 6852, 3, 0, 3814, 8, 718, 257, 8181, 6, 0, 3828, 2, 58, 0, 155, 21, 718, 7653, 17, 1, 8026, 1390, 3, 3, 3, 58, 1406, 29, 179, 21, 0, 3, 0, 2313, 1587, 13, 8, 43, 11, 162, 426, 3539, 0, 6, 86, 363, 4, 1, 759, 149, 8509, 1, 585, 17, 718, 7647, 0, 3, 23, 614, 13, 1, 759, 149, 0, 17, 7826, 6, 200, 363, 3, 23, 30, 8, 4422, 440, 6, 565, 2926, 4, 1, 759, 149, 0, 6, 350, 2, 17, 0, 5842, 3, 23, 3814, 31, 7283, 7939, 6770, 20, 8, 1064, 0, 3, 23, 2666, 13, 1, 7588, 4148, 6, 7963, 8, 7939, 13, 759, 1169, 8, 346, 26, 8, 4505, 1398, 3, 0, 306, 1863, 193, 306, 12, 416, 3441, 96, 53, 306, 1015, 12, 235, 26, 362, 6851, 38, 74, 33, 5442, 13, 360, 58, 33, 7356, 24, 1177, 306, 90, 11329, 5, 1851, 306, 1, 5647, 7, 3889, 74, 5, 1272, 148, 306, 1863, 193, 306, 12, 416, 1837, 3, 0]
    train_data = tensor(numpy.array(train_data, dtype=numpy.int64))
    train_data = train_data.to(torch.int64).to(device)
    bptt = 35
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    ntokens, emsize, nhead, d_hid, nlayers, dropout = 12455, 200, 2, 200, 2, 0.2
    model = ORTModule(
        TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)

    for epoch in range(1, 2):
        model.train()  # turn on train mode

        num_batches = len(train_data) // bptt
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:  # only on last batch
                src_mask = src_mask[:batch_size, :batch_size]
            try:
                output = model(data, src_mask)
            except RuntimeError as e:
                if torch_forward:
                    raise AssertionError("Fallback failed: %r." % e)
            if not torch_forward:
                raise AssertionError("Fallback was not used but policy is %r." % policy)
            nrows = min(ntokens, targets.shape[0])
            loss = criterion(output.view(nrows, -1), targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            break

    model_copied = copy.deepcopy(model)
    assert model_copied is not model_copied
    pkl = pickle.dump(model)
    assert pkl is not None
