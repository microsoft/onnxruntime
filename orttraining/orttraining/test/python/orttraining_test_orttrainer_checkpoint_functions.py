import pytest
from unittest.mock import patch, Mock
from orttraining_test_orttrainer_frontend import _load_pytorch_transformer_model
from onnxruntime.training import amp, checkpoint, optim, orttrainer
import numpy as np
import onnx
import torch

# Helper functions

def _create_trainer(zero_enabled=False):
    """Cerates a simple ORTTrainer for ORTTrainer functional tests"""

    device = 'cuda'
    optim_config = optim.LambConfig(lr=0.1)
    opts = {
                'device' : {'id' : device},
                'debug' : {'deterministic_compute': True}
            }
    if zero_enabled:
        opts['distributed'] = {
                'world_rank' : 0,
                'world_size' : 1,
                'allreduce_post_accumulation' : True,
                'deepspeed_zero_optimization':
                {
                    'stage': 1
                }
            }
    model, model_desc, loss_fn, batcher_fn, train_data, _, _ = _load_pytorch_transformer_model(device)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, loss_fn=loss_fn, options=orttrainer.ORTTrainerOptions(opts))

    return trainer

class _training_session_mock(object):
    """Mock object for the ORTTrainer _training_session member"""

    def __init__(self, model_states, optimizer_states, partition_info):
        self.model_states = model_states
        self.optimizer_states = optimizer_states
        self.partition_info = partition_info

    def get_model_state(self, include_mixed_precision_weights=False):
        return self.model_states

    def get_optimizer_state(self):
        return self.optimizer_states

    def get_partition_info_map(self):
        return self.partition_info

def _get_load_state_dict_strict_error_arguments():
    """Build parameterized list of arguments to test strict loading of the state dictionary"""

    training_session_state_dict = {
        'model': {
            'fp32': {
                'a': np.arange(5),
                'b': np.arange(7)
            }
        },
        'optimizer': {
            'a': {
                'Moment_1': np.arange(5),
                'Moment_2': np.arange(7)
            },
            'shared_optimizer_state': {
                'step': np.arange(5)
            }
        }
    }

    # input state dictionaries
    model_key_missing = {'optimizer': {}}
    optimizer_key_missing = {'model': {}}
    precision_key_missing = {'model': {}, 'optimizer': {}}
    precision_key_unexpected = {'model': {'fp32': {}, 'fp16': {}}, 'optimizer': {}}
    model_state_key_missing = {'model': {'fp32': {}}, 'optimizer': {}}
    model_state_key_unexpected = {'model': {'fp32': {'a': 2, 'b': 3, 'c': 4}}, 'optimizer': {}}
    optimizer_model_state_key_missing = {'model': {'fp32': {'a': 2, 'b': 3}}, 'optimizer': {}}
    optimizer_model_state_key_unexpected = {'model': {'fp32': {'a': 2, 'b': 3}}, 'optimizer': \
        {'a': {}, 'shared_optimizer_state': {}, 'b': {}}}
    optimizer_state_key_missing = {'model': {'fp32': {'a': 2, 'b': 3}}, 'optimizer': \
        {'a': {}, 'shared_optimizer_state': {'step': np.arange(5)}}}
    optimizer_state_key_unexpected = {'model': {'fp32': {'a': 2, 'b': 3}}, 'optimizer': \
        {'a': {'Moment_1': np.arange(5), 'Moment_2': np.arange(7)}, 'shared_optimizer_state': {'step': np.arange(5), 'another_step': np.arange(1)}}}

    input_arguments = [
        (training_session_state_dict, model_key_missing, ['model']),
        (training_session_state_dict, optimizer_key_missing, ['optimizer']),
        (training_session_state_dict, precision_key_missing, ['fp32']),
        (training_session_state_dict, precision_key_unexpected, ['fp16']),
        (training_session_state_dict, model_state_key_missing, ['a', 'b']),
        (training_session_state_dict, model_state_key_unexpected, ['c']),
        (training_session_state_dict, optimizer_model_state_key_missing, ['a', 'shared_optimizer_state']),
        (training_session_state_dict, optimizer_model_state_key_unexpected, ['b']),
        (training_session_state_dict, optimizer_state_key_missing, ['Moment_1', 'Moment_2']),
        (training_session_state_dict, optimizer_state_key_unexpected, ['another_step'])
    ]

    return input_arguments

# Tests

def test_empty_state_dict_when_training_session_uninitialized():
    trainer = _create_trainer()
    with pytest.warns(UserWarning) as user_warning:
        state_dict = trainer.state_dict()

    assert len(state_dict.keys()) == 0
    assert user_warning[0].message.args[0] == "ONNX Runtime training session is not initialized yet. " \
    "Please run train_step or eval_step at least once before calling ORTTrainer.state_dict()."

@patch('onnx.ModelProto')
def test_training_session_provides_empty_model_states(onnx_model_mock):
    trainer = _create_trainer()
    training_session_mock = _training_session_mock({}, {}, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict()
    assert len(state_dict['model'].keys()) == 0

@patch('onnx.ModelProto')
def test_training_session_provides_model_states(onnx_model_mock):
    trainer = _create_trainer()
    model_states = {
        'fp32': {
            'a': np.arange(5),
            'b': np.arange(7)
        }
    }
    training_session_mock = _training_session_mock(model_states, {}, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict()
    assert torch.all(torch.eq(state_dict['model']['fp32']['a'], torch.tensor(np.arange(5))))
    assert torch.all(torch.eq(state_dict['model']['fp32']['b'], torch.tensor(np.arange(7))))

@patch('onnx.ModelProto')
def test_onnx_graph_provides_frozen_model_states(onnx_model_mock):
    trainer = _create_trainer()
    model_states = {
        'fp32': {
            'a': np.arange(5),
            'b': np.arange(7)
        }
    }
    training_session_mock = _training_session_mock(model_states, {}, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()
    trainer.options.utils.frozen_weights = ['a_frozen_weight', 'a_float16_weight']
    trainer._onnx_model.graph.initializer = [
        onnx.numpy_helper.from_array(np.array([1, 2, 3], dtype=np.float32), 'a_frozen_weight'),
        onnx.numpy_helper.from_array(np.array([4, 5, 6], dtype=np.float32), 'a_non_fronzen_weight'),
        onnx.numpy_helper.from_array(np.array([7, 8, 9], dtype=np.float16), 'a_float16_weight')
    ]

    state_dict = trainer.state_dict()
    assert torch.all(torch.eq(state_dict['model']['fp32']['a'], torch.tensor(np.arange(5))))
    assert torch.all(torch.eq(state_dict['model']['fp32']['b'], torch.tensor(np.arange(7))))
    assert torch.all(torch.eq(state_dict['model']['fp32']['a_frozen_weight'], torch.tensor(np.array([1, 2, 3], dtype=np.float32))))
    assert 'a_non_fronzen_weight' not in state_dict['model']['fp32']
    assert 'a_float16_weight' not in state_dict['model']['fp32']

@patch('onnx.ModelProto')
def test_training_session_provides_empty_optimizer_states(onnx_model_mock):
    trainer = _create_trainer()
    training_session_mock = _training_session_mock({}, {}, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict()
    assert len(state_dict['optimizer'].keys()) == 0

@patch('onnx.ModelProto')
def test_training_session_provides_optimizer_states(onnx_model_mock):
    trainer = _create_trainer()
    optimizer_states = {
        'model_weight': {
            'Moment_1': np.arange(5),
            'Moment_2': np.arange(7)
        },
        'shared_optimizer_state': {
            'step': np.arange(1)
        }
    }
    training_session_mock = _training_session_mock({}, optimizer_states, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict()
    assert torch.all(torch.eq(state_dict['optimizer']['model_weight']['Moment_1'], torch.tensor(np.arange(5))))
    assert torch.all(torch.eq(state_dict['optimizer']['model_weight']['Moment_2'], torch.tensor(np.arange(7))))
    assert torch.all(torch.eq(state_dict['optimizer']['shared_optimizer_state']['step'], torch.tensor(np.arange(1))))

@patch('onnx.ModelProto')
def test_training_session_provides_empty_partition_info_map(onnx_model_mock):
    trainer = _create_trainer(zero_enabled=True)
    training_session_mock = _training_session_mock({}, {}, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict()
    assert len(state_dict['partition_info'].keys()) == 0

@patch('onnx.ModelProto')
def test_training_session_provides_partition_info_map(onnx_model_mock):
    trainer = _create_trainer(zero_enabled=True)
    partition_info = {
        'a': {
            'original_dim': [1, 2, 3]
        }
    }
    training_session_mock = _training_session_mock({}, {}, partition_info)
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict()
    assert state_dict['partition_info']['a']['original_dim'] == [1, 2, 3]

@patch('onnx.ModelProto')
def test_training_session_provides_all_states(onnx_model_mock):
    trainer = _create_trainer(zero_enabled=True)
    model_states = {
        'fp32': {
            'a': np.arange(5),
            'b': np.arange(7)
        }
    }
    optimizer_states = {
        'model_weight': {
            'Moment_1': np.arange(5),
            'Moment_2': np.arange(7)
        },
        'shared_optimizer_state': {
            'step': np.arange(1)
        }
    }
    partition_info = {
        'a': {
            'original_dim': [1, 2, 3]
        }
    }
    training_session_mock = _training_session_mock(model_states, optimizer_states, partition_info)
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict()
    assert torch.all(torch.eq(state_dict['model']['fp32']['a'], torch.tensor(np.arange(5))))
    assert torch.all(torch.eq(state_dict['model']['fp32']['b'], torch.tensor(np.arange(7))))
    assert torch.all(torch.eq(state_dict['optimizer']['model_weight']['Moment_1'], torch.tensor(np.arange(5))))
    assert torch.all(torch.eq(state_dict['optimizer']['model_weight']['Moment_2'], torch.tensor(np.arange(7))))
    assert torch.all(torch.eq(state_dict['optimizer']['shared_optimizer_state']['step'], torch.tensor(np.arange(1))))
    assert state_dict['partition_info']['a']['original_dim'] == [1, 2, 3]

def test_load_state_dict_holds_when_training_session_not_initialized():
    trainer = _create_trainer()
    state_dict = {
        'model': {
            'fp32': {
                'a': np.arange(5),
                'b': np.arange(7)
            }
        },
        'optimizer': {
            'a': {
                'Moment_1': np.arange(5),
                'Moment_2': np.arange(7)
            },
            'shared_optimizer_state': {
                'step': np.arange(5)
            }
        }
    }
    assert not trainer._load_state_dict
    state_dict = trainer.load_state_dict(state_dict)
    assert trainer._load_state_dict

@pytest.mark.parametrize("state_dict, input_state_dict, error_keys", _get_load_state_dict_strict_error_arguments())
def test_load_state_dict_errors_when_model_key_missing(state_dict, input_state_dict, error_keys):
    trainer = _create_trainer()
    trainer._training_session = _training_session_mock({}, {}, {})
    trainer.state_dict = Mock(return_value=state_dict)
    with pytest.raises(RuntimeError) as runtime_error:
        trainer.load_state_dict(input_state_dict)

    assert any(key in str(runtime_error.value) for key in error_keys)

@patch('onnx.ModelProto')
def test_load_state_dict_loads_the_states_and_inits_training_session(onnx_model_mock):
    trainer = _create_trainer()
    training_session_state_dict = {
        'model': {
            'fp32': {
                'a': np.arange(5),
                'b': np.arange(7)
            }
        },
        'optimizer': {
            'a': {
                'Moment_1': np.arange(5),
                'Moment_2': np.arange(7)
            },
            'shared_optimizer_state': {
                'step': np.arange(1)
            }
        }
    }

    input_state_dict = {
        'model': {
            'fp32': {
                'a': torch.tensor(np.array([1, 2])),
                'b': torch.tensor(np.array([3, 4]))
            }
        },
        'optimizer': {
            'a': {
                'Moment_1': torch.tensor(np.array([5, 6])),
                'Moment_2': torch.tensor(np.array([7, 8]))
            },
            'shared_optimizer_state': {
                'step': torch.tensor(np.array([9]))
            }
        }
    }
    trainer._training_session = _training_session_mock({}, {}, {})
    trainer.state_dict = Mock(return_value=training_session_state_dict)
    trainer._onnx_model = onnx_model_mock()
    trainer._onnx_model.graph.initializer = [
        onnx.numpy_helper.from_array(np.arange(20, dtype=np.float32), 'a'),
        onnx.numpy_helper.from_array(np.arange(25, dtype=np.float32), 'b')
    ]
    trainer._update_onnx_model_initializers = Mock()
    trainer._init_session = Mock()

    trainer.load_state_dict(input_state_dict)

    loaded_initializers, _ = trainer._update_onnx_model_initializers.call_args
    state_dict_to_load, _ = trainer._init_session.call_args

    assert 'a' in loaded_initializers[0]
    assert (loaded_initializers[0]['a'] == np.array([1, 2])).all()
    assert 'b' in loaded_initializers[0]
    assert (loaded_initializers[0]['b'] == np.array([3, 4])).all()

    assert (state_dict_to_load[0]['optimizer']['a']['Moment_1'] ==  np.array([5, 6])).all()
    assert (state_dict_to_load[0]['optimizer']['a']['Moment_2'] ==  np.array([7, 8])).all()
    assert (state_dict_to_load[0]['optimizer']['shared_optimizer_state']['step'] ==  np.array([9])).all()
