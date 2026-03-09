# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from unittest.mock import patch

import pytest
from packaging import version

from olive.telemetry.telemetry import Telemetry
from test.utils import create_onnx_model_file, delete_onnx_model_files


@pytest.fixture(scope="session", autouse=True)
def setup_onnx_model(request, tmp_path_factory):
    cache_path = tmp_path_factory.mktemp("transformers_cache")
    import transformers

    # we cannot use os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
    # because the TRANSFORMERS_CACHE is loaded when importing transformers
    transformers.utils.hub.TRANSFORMERS_CACHE = str(cache_path)

    from datasets import disable_caching

    disable_caching()
    create_onnx_model_file()
    yield
    delete_onnx_model_files()
    shutil.rmtree(cache_path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def maybe_patch_inc():
    import peft

    if version.parse(peft.__version__) >= version.parse("0.16.0"):
        # peft 0.16.0+ has a new dispatcher for inc which imports missing dependencies
        with patch("peft.tuners.lora.inc.is_inc_available", new=lambda: False):
            yield
    else:
        yield


@pytest.fixture(scope="session", autouse=True)
def disable_telemetry():
    Telemetry().disable_telemetry()
