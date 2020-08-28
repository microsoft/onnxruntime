import inspect
import onnx
import os
import pytest
import torch

from numpy.testing import assert_allclose

from onnxruntime import set_seed
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.experimental import _utils, amp, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options
import _test_helpers


###############################################################################
# Helper functions ############################################################
###############################################################################


def _load_pytorch_transformer_model(device, dynamic_axes=False, legacy_api=False):
    # Loads external Pytorch TransformerModel into utils
    pytorch_transformer_path = os.path.join('..', '..', '..', 'samples', 'python', 'pytorch_transformer')
    pt_model_path = os.path.join(pytorch_transformer_path, 'pt_model.py')
    pt_model = _utils.import_module_from_file(pt_model_path)
    ort_utils_path = os.path.join(pytorch_transformer_path, 'ort_utils.py')
    ort_utils = _utils.import_module_from_file(ort_utils_path)
    utils_path = os.path.join(pytorch_transformer_path, 'utils.py')
    utils = _utils.import_module_from_file(utils_path)

    # Modeling
    model = pt_model.TransformerModel(28785, 200, 2, 200, 2, 0.2).to(device)
    my_loss = ort_utils.my_loss
    if legacy_api:
        if dynamic_axes:
            model_desc = ort_utils.legacy_transformer_model_description_dynamic_axes()
        else:
            model_desc = ort_utils.legacy_transformer_model_description()
    else:
        if dynamic_axes:
            model_desc = ort_utils.transformer_model_description_dynamic_axes()
        else:
            model_desc = ort_utils.transformer_model_description()


    # Preparing data
    train_data, val_data, test_data = utils.prepare_data(device, 20, 20)
    return model, model_desc, my_loss, utils.get_batch, train_data, val_data, test_data


###############################################################################
# Testing starts here #########################################################
###############################################################################


@pytest.mark.parametrize("seed, device", [
    (0, 'cpu'),
    (24, 'cuda')
])
def testORTDeterministicCompute(seed, device):
    # Common setup
    optim_config = optim.LambConfig()
    opts = orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True,
            'check_model_export': True,
        },
        'device' : {
            'id' : device,
        }
    })

    # Setup for the first ORTTRainer run
    torch.manual_seed(seed)
    set_seed(seed)
    model, model_desc, my_loss, batcher_fn, train_data, val_data, _ = _load_pytorch_transformer_model(device)
    first_trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=my_loss, options=opts)
    data, targets = batcher_fn(train_data, 0)
    _ = first_trainer.train_step(data, targets)
    assert first_trainer._onnx_model is not None



