import pytest
import torch

from onnxruntime.capi.training import orttrainer_options as orttrainer_options
from onnxruntime.capi.training import model_desc_validation as md_val


@pytest.mark.parametrize("test_input", [
    ({}),
    ({'batch': {},
      'device': {},
      'distributed': {},
      'mixed_precision': {},
      'utils': {},
      '_internal_use': {}})
])
def testORTTrainerOptionsDefaultValues(test_input):
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
            'deepspeed_zero_stage': 0,
            'enable_adasum': False
        },
        'lr_scheduler': None,
        'mixed_precision': {
            'enabled': False,
            'loss_scaler': None
        },
        'utils': {
            'frozen_weights': [],
            'grad_norm_clip': False
        },
        '_internal_use': {
            'enable_internal_postprocess': True,
            'extra_postprocess': None
        }
    }

    actual_values = orttrainer_options.ORTTrainerOptions(test_input)
    assert actual_values._validated_opts == expected_values


def testORTTrainerOptionsInvalidMixedPrecisionEnabledSchema():
    '''Test an invalid input based on schema validation error message'''

    expected_msg = "Invalid options: {'mixed_precision': [{'enabled': ['must be of boolean type']}]}"
    with pytest.raises(ValueError) as e:
        orttrainer_options.ORTTrainerOptions(
            {'mixed_precision': {'enabled': 1}})
    assert str(e.value) == expected_msg


@pytest.mark.parametrize("test_input", [
    ({'inputs': [('in0', [])],
      'outputs': [('out0', []), ('out1', [])]}),
    ({'inputs': [('in0', ['batch', 2, 3])],
      'outputs': [('out0', [], True)]}),
    ({'inputs': [('in0', []), ('in1', [1]), ('in2', [1, 2]), ('in3', [1000, 'dyn_ax1']), ('in4', ['dyn_ax1', 'dyn_ax2', 'dyn_ax3'])],
      'outputs': [('out0', [], True), ('out1', [1], False), ('out2', [1, 'dyn_ax1', 3])]})
])
def testORTTrainerModelDescValidSchemas(test_input):
    r''' Test different ways of using default values for incomplete input'''
    md_val._ORTTrainerModelDesc(test_input)


@pytest.mark.parametrize("test_input,error_msg", [
    ({'inputs': [(True, [])],
      'outputs': [(True, [])]},
     "Invalid model_desc: {'inputs': [{0: ['the first element of the tuple (aka name) must be a string']}], 'outputs': [{0: ['the first element of the tuple (aka name) must be a string']}]}"),
    ({'inputs': [('in1', None)],
      'outputs': [('out1', None)]},
     "Invalid model_desc: {'inputs': [{0: ['the second element of the tuple (aka shape) must be a list']}], 'outputs': [{0: ['the second element of the tuple (aka shape) must be a list']}]}"),
    ({'inputs': [('in1', [])],
      'outputs': [('out1', [], None)]},
     "Invalid model_desc: {'outputs': [{0: ['the third element of the tuple (aka is_loss) must be a boolean']}]}"),
    ({'inputs': [('in1', [True])],
      'outputs': [('out1', [True])]},
     "Invalid model_desc: {'inputs': [{0: ['each shape must be either a string or integer']}], 'outputs': [{0: ['each shape must be either a string or integer']}]}"),
    ({'inputs': [('in1', [])],
      'outputs': [('out1', [], True), ('out2', [], True)]},
     "Invalid model_desc: {'outputs': [{1: ['only one is_loss can bet set to True']}]}"),
])
def testORTTrainerModelDescInvalidSchemas(test_input, error_msg):
    r''' Test different ways of using default values for incomplete input'''
    with pytest.raises(ValueError) as e:
        md_val._ORTTrainerModelDesc(test_input)
    assert str(e.value) == error_msg
