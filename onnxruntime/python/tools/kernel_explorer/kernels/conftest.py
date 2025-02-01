# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_once():

    print("Setup before all tests.")
    from kernel_explorer import onnxruntime_pybind11_state
    onnxruntime_pybind11_state.set_default_logger_severity(0)
    onnxruntime_pybind11_state.set_default_logger_verbosity(0)

    yield
    print("Teardown after all tests.")
