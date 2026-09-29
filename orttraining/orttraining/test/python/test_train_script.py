# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import shlex
from pathlib import Path
from unittest.mock import patch


def load_train_module():
    train_script = Path(__file__).resolve().parents[4] / "orttraining/tools/scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_script", train_script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_script_does_not_execute_shell_command_on_import():
    with (
        patch("os.system", side_effect=AssertionError("os.system should not be used")),
        patch("subprocess.run", side_effect=AssertionError("subprocess.run should not be used on import")),
    ):
        module = load_train_module()

    assert hasattr(module, "main")
    assert callable(module.main)


def test_train_main_uses_argument_list_for_subprocess():
    module = load_train_module()

    with patch("subprocess.run") as mock_run, patch("builtins.print") as mock_print:
        module.main(["--foo", "bar; id"])

    mock_run.assert_called_once_with(
        ["/workspace/onnxruntime_training_bert", "--foo", "bar; id"],
        check=True,
    )
    mock_print.assert_called_once_with(shlex.join(["/workspace/onnxruntime_training_bert", "--foo", "bar; id"]))
