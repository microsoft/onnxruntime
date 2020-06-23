import pytest
import torch

from onnxruntime.capi.training import pytorch_trainer_options as pt_options


@pytest.mark.parametrize("test_input", [
    ({}),
    ({'batch': {},
      'device': {},
      'distributed': {},
      'mixed_precision': {},
      'utils': {},
      '_internal_use': {}})
])
def testPytorchTrainerOptionsDefaultValues(test_input):
    ''' Test different ways of using default values for incomplete input'''

    expected_values = {
        'batch': {
            'gradient_accumulation_steps': 0
        },
        'device': {
            'id': None,
            'mem_limit': 0
        },
        'distributed': {
            'world_rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'allreduce_post_accumulation': False,
            'enable_partition_optimizer': False,
            'enable_adasum': False
        },
        'lr_scheduler': None,
        'mixed_precision': {
            'enabled': False,
            'loss_scaler': None
        },
        'utils': {
            'grad_norm_clip': False
        },
        '_internal_use': {
            'frozen_weights': [],
            'enable_internal_postprocess': True,
            'extra_postprocess': None
        }
    }

    actual_values = pt_options.PytorchTrainerOptions(test_input)
    assert actual_values._validated_opts == expected_values


def testPytorchTrainerOptionsInvalidMixedPrecisionEnabledSchema():
    '''Test an invalid input based on schema validation error message'''

    expected_msg = 'must be of boolean type'
    actual_values = pt_options.PytorchTrainerOptions(
        {'mixed_precision': {'enabled': 1}})
    assert actual_values.mixed_precision[0].enabled[0] == expected_msg
