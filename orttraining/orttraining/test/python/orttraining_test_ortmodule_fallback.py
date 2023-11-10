# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# orttraining_test_ortmodule_api.py

import copy
import itertools
import math
import os
import warnings

import _test_helpers
import numpy as np
import pytest
import torch
from _orttraining_ortmodule_models import (
    MyCustomClassInputNet,
    NeuralNetCustomClassOutput,
    NeuralNetSinglePositionalArgument,
)

from onnxruntime.training.ortmodule import ORTMODULE_TORCH_CPP_DIR, ORTModule, _fallback
from onnxruntime.training.ortmodule.torch_cpp_extensions import is_installed as is_torch_cpp_extensions_installed

# PyTorch model definitions for tests


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_forward(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_FORCE_TORCH_FORWARD policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = "FALLBACK_FORCE_TORCH_FORWARD"
        else:
            policy = "FALLBACK_UNSUPPORTED_DEVICE"
    else:
        policy = "FALLBACK_DISABLE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

    from dataclasses import dataclass

    @dataclass
    class Point:
        x: int
        y: int

    class UnsupportedInputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

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
                    assert (
                        ort_model._torch_module._execution_manager(is_training=is_training)._fallback_manager._exception
                        is not None
                    )
                ort_out = ort_model(inputs)
                pt_out = pt_model(inputs)
                assert ort_out == pt_out
            else:
                with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                    ort_model(inputs)
                assert "ORTModule fails to extract schema from data" in str(type_error.value)
        else:
            with pytest.raises(_fallback.ORTModuleFallbackException) as type_error:
                ort_model(inputs)
            assert "ORTModule fails to extract schema from data" in str(type_error.value)


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_device__multiple(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_DEVICE policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DATA) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = "FALLBACK_UNSUPPORTED_DEVICE"
        else:
            policy = "FALLBACK_UNSUPPORTED_DATA"
    else:
        policy = "FALLBACK_DISABLE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

    class ManyDevicesNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net1 = torch.nn.Linear(10, 10).to("cuda:0")
            self.relu = torch.nn.ReLU()
            self.net2 = torch.nn.Linear(10, 5).to("cpu")

        def forward(self, x):
            x = self.relu(self.net1(x.to("cuda:0")))
            return self.net2(x.to("cpu"))

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


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_device__mismatch(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_DEVICE policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DATA) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = "FALLBACK_UNSUPPORTED_DEVICE"
        else:
            policy = "FALLBACK_UNSUPPORTED_DATA"
    else:
        policy = "FALLBACK_DISABLE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)
    os.environ["ORTMODULE_SKIPCHECK_POLICY"] = "SKIP_CHECK_DISABLED"

    data_device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10  # noqa: N806

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
                assert (
                    "Tensor for argument #1 'self' is on CPU, but expected them to be on GPU (while checking arguments for addmm)"
                    in str(e.value)
                ) or (
                    "Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!"
                    in str(e.value)
                )
            else:
                with pytest.raises(_fallback.ORTModuleDeviceException) as e:
                    ort_model(inputs)
                assert (
                    f"Input argument to forward found on device {input_device}, "
                    f"but expected it to be on module device {ort_model_device}." in str(e.value)
                )
        else:
            with pytest.raises(_fallback.ORTModuleDeviceException) as e:
                ort_model(inputs)
            assert (
                f"Input argument to forward found on device {input_device}, "
                f"but expected it to be on module device {ort_model_device}." in str(e.value)
            )

    del os.environ["ORTMODULE_SKIPCHECK_POLICY"]


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_output(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_DATA policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = "FALLBACK_UNSUPPORTED_DATA"
        else:
            policy = "FALLBACK_UNSUPPORTED_DEVICE"
    else:
        policy = "FALLBACK_DISABLE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10  # noqa: N806
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
                    assert (
                        ort_model._torch_module._execution_manager(is_training=is_training)._fallback_manager._exception
                        is not None
                    )
                ort_out = ort_model(x, y, z)
                pt_out = pt_model(x, y, z)
                _test_helpers.assert_values_are_close(ort_out.out1, pt_out.out1, rtol=0, atol=0)
                _test_helpers.assert_values_are_close(ort_out.out2, pt_out.out2, rtol=0, atol=0)
                _test_helpers.assert_values_are_close(ort_out.out3, pt_out.out3, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleIOError) as runtime_error:
                    ort_model(x, y, z)
                assert "ORTModule does not support the following model output type" in str(runtime_error.value)
        else:
            with pytest.raises(_fallback.ORTModuleIOError) as runtime_error:
                ort_model(x, y, z)
            assert "ORTModule does not support the following model output type" in str(runtime_error.value)


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_input(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_DATA policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = "FALLBACK_UNSUPPORTED_DATA"
        else:
            policy = "FALLBACK_UNSUPPORTED_DEVICE"
    else:
        policy = "FALLBACK_DISABLE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

    pt_model = MyCustomClassInputNet()
    ort_model = ORTModule(copy.deepcopy(pt_model))
    inputs = torch.randn(1, 2)

    class CustomClass:
        def __init__(self, x):
            self.x = x

    ort_model.train(is_training)
    pt_model.train(is_training)

    for i in range(3):
        if fallback_enabled:
            if matching_policy:
                if i > 0 and persist_fallback:
                    assert (
                        ort_model._torch_module._execution_manager(is_training=is_training)._fallback_manager._exception
                        is not None
                    )
                ort_out = ort_model(inputs, CustomClass(1))
                pt_out = pt_model(inputs, CustomClass(1))
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)
            else:
                with pytest.raises(_fallback.ORTModuleIOError) as ex_info:
                    _ = ort_model(torch.randn(1, 2), CustomClass(1))
                assert (
                    "ORTModule fails to extract schema from data: "
                    "Unsupported flatten data type: "
                    "<class 'orttraining_test_ortmodule_fallback."
                    "test_ortmodule_fallback_input.<locals>.CustomClass'>" in str(ex_info.value)
                )
        else:
            with pytest.raises(_fallback.ORTModuleIOError) as ex_info:
                _ = ort_model(torch.randn(1, 2), CustomClass(1))
            assert (
                "ORTModule fails to extract schema from data: "
                "Unsupported flatten data type: "
                "<class 'orttraining_test_ortmodule_fallback."
                "test_ortmodule_fallback_input.<locals>.CustomClass'>" in str(ex_info.value)
            )


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_torch_model(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = "FALLBACK_UNSUPPORTED_TORCH_MODEL"
        else:
            policy = "FALLBACK_UNSUPPORTED_DEVICE"
    else:
        policy = "FALLBACK_DISABLE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

    device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10  # noqa: N806
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
                _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=1e-3, atol=1e-6)
            else:
                with pytest.raises(_fallback.ORTModuleTorchModelException) as ex_info:
                    ort_model = ORTModule(pt_model)
                assert "ORTModule is not compatible with torch.nn.DataParallel" in str(ex_info.value)
        else:
            with pytest.raises(_fallback.ORTModuleTorchModelException) as ex_info:
                # Initialize with fallback policy because Exception will happen during __init__
                ort_model = ORTModule(pt_model)
            assert "ORTModule is not compatible with torch.nn.DataParallel" in str(ex_info.value)


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_init__torch_version(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    from packaging import version

    from onnxruntime.training.ortmodule import MINIMUM_RUNTIME_PYTORCH_VERSION_STR

    runtime_pytorch_version = version.parse(torch.__version__.split("+")[0])
    minimum_runtime_pytorch_version = version.parse(MINIMUM_RUNTIME_PYTORCH_VERSION_STR)
    if runtime_pytorch_version < minimum_runtime_pytorch_version:
        if fallback_enabled:
            if matching_policy:
                policy = "FALLBACK_BAD_INITIALIZATION"
            else:
                policy = "FALLBACK_UNSUPPORTED_DEVICE"
        else:
            policy = "FALLBACK_DISABLE"
        os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
        os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

        device = "cuda"
        N, D_in, H, D_out = 64, 784, 500, 10  # noqa: N806
        x = torch.randn(N, D_in, device=device)

        pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out).to(device)

        for _i in range(3):
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
                        ex_info.value
                    )
            else:
                with pytest.raises(_fallback.ORTModuleInitException) as ex_info:
                    # Initialize with fallback policy because Exception will happen during __init__
                    ort_model = ORTModule(pt_model)
                assert "ONNX Runtime ORTModule frontend requires PyTorch version greater or equal to" in str(
                    ex_info.value
                )
    else:
        warnings.warn(
            "Skipping test_ortmodule_fallback_torch_version."
            f" It requires PyTorch prior to {MINIMUM_RUNTIME_PYTORCH_VERSION_STR}"
        )


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_init__missing_cpp_extensions(
    is_training, fallback_enabled, matching_policy, persist_fallback
):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_TORCH_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if is_torch_cpp_extensions_installed(ORTMODULE_TORCH_CPP_DIR):
        warnings.warn(
            "Skipping test_ortmodule_fallback_init__missing_cpp_extensions."
            " It requires PyTorch CPP extensions to be missing"
        )
    else:
        if fallback_enabled:
            if matching_policy:
                policy = "FALLBACK_BAD_INITIALIZATION"
            else:
                policy = "FALLBACK_UNSUPPORTED_DEVICE"
        else:
            policy = "FALLBACK_DISABLE"
        os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
        os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

        device = "cuda"
        N, D_in, H, D_out = 64, 784, 500, 10  # noqa: N806
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


@pytest.mark.parametrize(
    "is_training,fallback_enabled,matching_policy,persist_fallback", list(itertools.product([True, False], repeat=4))
)
def test_ortmodule_fallback_onnx_model__missing_op(is_training, fallback_enabled, matching_policy, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend
    # matching_policy: True matches FALLBACK_UNSUPPORTED_ONNX_MODEL policy to ORTModuleDeviceException exception.
    #   Otherwise, an incorrect policy (FALLBACK_UNSUPPORTED_DEVICE) is used to verify that the fallback does not happen

    if fallback_enabled:
        if matching_policy:
            policy = "FALLBACK_UNSUPPORTED_ONNX_MODEL"
        else:
            policy = "FALLBACK_UNSUPPORTED_DEVICE"
    else:
        policy = "FALLBACK_DISABLE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

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
                    assert (
                        ort_model._torch_module._execution_manager(is_training=is_training)._fallback_manager._exception
                        is not None
                    )
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


@pytest.mark.parametrize("is_training,persist_fallback", list(itertools.product([True, False], repeat=2)))
def test_ortmodule_fallback_warn_message(is_training, persist_fallback, caplog):
    # is_training: True for torch.nn.Module training model, eval mode otherwise

    policy = "FALLBACK_UNSUPPORTED_DEVICE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)
    os.environ["ORTMODULE_SKIPCHECK_POLICY"] = "SKIP_CHECK_DISABLED"

    data_device = "cuda"
    N, D_in, H, D_out = 64, 784, 500, 10  # noqa: N806

    pt_model = NeuralNetSinglePositionalArgument(D_in, H, D_out)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    pt_model.train(is_training)
    ort_model.train(is_training)
    # For initial model export, use same device for data and model so that PyTorch model can be traced during export
    _ = ort_model(torch.randn(N, D_in))

    # Use data in different device for testing
    inputs = torch.randn(N, D_in, device=data_device)

    for i in range(3):
        # The run MUST fail no matter with ORT or PyTorch, so we catch the RuntimeError here in case it breaks the test.
        # Be noted, the logs will be caught by caplog.
        with pytest.raises(RuntimeError):
            ort_model(inputs)
        # For retries, the warn message will always be logged
        if not persist_fallback:
            if i == 0:
                # For the first time, run ORTModule, feature map is logged as warning
                # And the fallback warning is logged.
                assert len(caplog.records) >= 2
            else:
                # For the other time, only the fallback warning is logged.
                assert len(caplog.records) == 1
            assert "Fallback to PyTorch due to exception" in caplog.records[-1].message
            caplog.clear()
            continue

        # If `retries` is not enabled, only log the warn message once
        if i == 0:
            # For the first time, run ORTModule, feature map is logged as warning
            # And the fallback warning is logged.
            assert len(caplog.records) >= 2
            assert "Fallback to PyTorch due to exception" in caplog.records[-1].message
            caplog.clear()
        else:
            # For the other time, no fallback warning will be logged because
            # we are running with PyTorch.
            assert len(caplog.records) == 0
            caplog.clear()

    del os.environ["ORTMODULE_SKIPCHECK_POLICY"]


@pytest.mark.parametrize("is_training,fallback_enabled,persist_fallback", [[True, True, True], [True, True, False]])
def test_ortmodule_fallback_duplicated_warn_message(is_training, fallback_enabled, persist_fallback):
    # This test is for the duplicated warn message from exceptions, e.g. FALLBACK_UNSUPPORTED_ONNX_MODEL
    # The fallback warn message will be logged when _raised_fallback_exception is False, vice versa
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # fallback_enabled: True PyTorch executes the forward graph instead of ORT backend

    policy = "FALLBACK_UNSUPPORTED_ONNX_MODEL"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)

    class CrossModule(torch.nn.Module):
        def forward(self, x, y):
            return torch.cross(x, y)

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    pt_model = CrossModule()
    ort_model = ORTModule(copy.deepcopy(pt_model))
    ort_model.train(is_training)
    pt_model.train(is_training)

    for i in range(5):
        if fallback_enabled:
            warn_message_logged = ort_model._torch_module._execution_manager(
                is_training=is_training
            )._fallback_manager._raised_fallback_exception
            if i > 0 and persist_fallback:
                assert warn_message_logged is True
            else:
                assert warn_message_logged is False
            pt_out = pt_model(x, y)
            ort_out = ort_model(x, y)
            _test_helpers.assert_values_are_close(ort_out, pt_out, rtol=0, atol=0)


# This test now results in a different error:
# torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::unflatten' to ONNX opset version 15 is not supported.
# Skip this test for pytorch 2.0 until fix identified.
@pytest.mark.xfail(reason="This test now results in an export error.")
@pytest.mark.parametrize("is_training,persist_fallback", list(itertools.product([True, False], repeat=2)))
def test_ortmodule_fallback_non_contiguous_tensors(is_training, persist_fallback):
    # is_training: True for torch.nn.Module training model, eval mode otherwise
    # Validate fix for issue: https://github.com/pytorch/ort/issues/92

    policy = "FALLBACK_UNSUPPORTED_DEVICE"
    os.environ["ORTMODULE_FALLBACK_POLICY"] = policy
    os.environ["ORTMODULE_FALLBACK_RETRY"] = str(not persist_fallback)
    os.environ["ORTMODULE_SKIPCHECK_POLICY"] = "SKIP_CHECK_DISABLED"

    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=dropout)
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[: x.size(0)]
            return self.dropout(x)

    class TransformerModel(torch.nn.Module):
        def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
            super().__init__()
            self.model_type = "Transformer"
            encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            self.encoder = torch.nn.Embedding(ntoken, d_model)
            self.d_model = d_model
            self.decoder = torch.nn.Linear(d_model, ntoken)
            self.init_weights()

        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, src, src_mask):
            src = self.encoder(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, src_mask)
            output = self.decoder(output)
            return output

    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i : i + seq_len]
        target = source[i + 1 : i + 1 + seq_len].reshape(-1)
        return data, target

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = np.random.randint(1, 12455, 1000)
    ends = np.random.randint(2, 20, 100).cumsum()
    ends = ends[ends < train_data.shape[0] - 2]
    train_data[ends] = 0
    train_data[-1] = 0

    train_data = torch.tensor(np.array(train_data, dtype=np.int64))
    train_data = train_data.to(torch.int64).to(device)
    bptt = 35
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    ntokens, emsize, nhead, d_hid, nlayers, dropout = 12455, 200, 2, 200, 2, 0.2
    pt_model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    model = ORTModule(pt_model).to(device)
    pt_model.train(is_training)
    model.train(is_training)
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)

    n_iter = 0
    for _epoch in range(1, 2):
        model.train()  # turn on train mode

        len(train_data) // bptt
        for _batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:  # only on last batch
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            nrows = min(ntokens, targets.shape[0])
            loss = criterion(output.view(nrows, -1), targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            n_iter += 1
            break

    assert n_iter > 0

    del os.environ["ORTMODULE_SKIPCHECK_POLICY"]
