import inspect  # noqa: F401
import os  # noqa: F401

import _test_helpers  # noqa: F401
import onnx  # noqa: F401
import pytest
import torch
import torchvision  # noqa: F401
from _test_commons import _load_pytorch_transformer_model
from numpy.testing import assert_allclose  # noqa: F401

from onnxruntime import set_seed
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription  # noqa: F401
from onnxruntime.capi.ort_trainer import LossScaler as Legacy_LossScaler  # noqa: F401
from onnxruntime.capi.ort_trainer import ModelDescription as Legacy_ModelDescription  # noqa: F401
from onnxruntime.capi.ort_trainer import ORTTrainer as Legacy_ORTTrainer  # noqa: F401
from onnxruntime.training import TrainStepInfo, _utils, amp  # noqa: F401
from onnxruntime.training import model_desc_validation as md_val  # noqa: F401
from onnxruntime.training import optim, orttrainer
from onnxruntime.training import orttrainer_options as orttrainer_options

###############################################################################
# Testing starts here #########################################################
###############################################################################


@pytest.mark.parametrize(
    "seed, device",
    [
        (24, "cuda"),
    ],
)
def testORTTransformerModelExport(seed, device):
    # Common setup
    optim_config = optim.LambConfig()
    opts = orttrainer.ORTTrainerOptions(
        {
            "debug": {
                "check_model_export": True,
            },
            "device": {
                "id": device,
            },
        }
    )

    # Setup for the first ORTTRainer run
    torch.manual_seed(seed)
    set_seed(seed)
    model, model_desc, my_loss, batcher_fn, train_data, val_data, _ = _load_pytorch_transformer_model(device)
    first_trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)
    data, targets = batcher_fn(train_data, 0)
    _ = first_trainer.train_step(data, targets)
    assert first_trainer._onnx_model is not None
