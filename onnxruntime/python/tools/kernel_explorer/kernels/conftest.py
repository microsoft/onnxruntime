# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_once():

    print("Setup before all tests.")
    from kernel_explorer import set_ort_severity
    set_ort_severity(3)

    yield

    print("Teardown after all tests.")
