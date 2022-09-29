import inspect
import onnx
import os
import pytest
import torch
import torchvision

from numpy.testing import assert_allclose

from onnxruntime import set_seed
from onnxruntime.capi.ort_trainer import (
    IODescription as Legacy_IODescription,
    ModelDescription as Legacy_ModelDescription,
    LossScaler as Legacy_LossScaler,
    ORTTrainer as Legacy_ORTTrainer,
)
from onnxruntime.training import (
    _utils,
    amp,
    optim,
    orttrainer,
    TrainStepInfo,
    model_desc_validation as md_val,
    orttrainer_options as orttrainer_options,
)

from _test_commons import _load_pytorch_transformer_model

import _test_helpers


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
