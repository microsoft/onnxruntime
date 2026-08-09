# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Configuration for pytest."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    # Suppress PyTorch TorchScript-based ONNX export deprecation warnings.
    # These come from torch.onnx.export when using the legacy exporter (dynamo=False).
    config.addinivalue_line(
        "filterwarnings",
        "ignore:You are using the legacy TorchScript-based ONNX export:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The feature will be removed:DeprecationWarning",
    )
    # Suppress TracerWarnings from PyTorch tracing (expected when tracing models with data-dependent control flow)
    config.addinivalue_line(
        "filterwarnings",
        "ignore::torch.jit.TracerWarning",
    )
    # Suppress UserWarning about dynamic axes validation from torch.onnx
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Provided key .* for dynamic axes is not a valid input/output name:UserWarning",
    )
    # Suppress UserWarning about ONNX inplace ops removal
    config.addinivalue_line(
        "filterwarnings",
        "ignore:ONNX Preprocess - Removing mutation from node:UserWarning",
    )
    # Suppress UserWarning about creating tensor from list of numpy ndarrays
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Creating a tensor from a list of numpy.ndarrays:UserWarning",
    )


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        # --slow given: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
