import pytest
from unittest.mock import patch, Mock
from _test_commons import _load_pytorch_transformer_model
from onnxruntime.training import amp, checkpoint, optim, orttrainer, _checkpoint_storage
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
                'horizontal_parallel_size' : 1,
                'data_parallel_size' : 1,
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
    """Return a list of tuples that can be used as parameters for test_load_state_dict_errors_when_model_key_missing

    Construct a list of tuples (training_session_state_dict, input_state_dict, error_arguments)
    The load_state_dict function will compare the two state dicts (training_session_state_dict, input_state_dict) and
    throw a runtime error with the missing/unexpected keys. The error arguments capture these missing/unexpected keys.
    """

    training_session_state_dict = {
        'model': {
            'full_precision': {
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
    precision_key_missing = {'model': {}, 'optimizer': {}}
    precision_key_unexpected = {'model': {'full_precision': {}, 'mixed_precision': {}}, 'optimizer': {}}
    model_state_key_missing = {'model': {'full_precision': {}}, 'optimizer': {}}
    model_state_key_unexpected = {'model': {'full_precision': {'a': 2, 'b': 3, 'c': 4}}, 'optimizer': {}}
    optimizer_model_state_key_missing = {'model': {'full_precision': {'a': 2, 'b': 3}}, 'optimizer': {}}
    optimizer_model_state_key_unexpected = {'model': {'full_precision': {'a': 2, 'b': 3}}, 'optimizer': \
        {'a': {}, 'shared_optimizer_state': {}, 'b': {}}}
    optimizer_state_key_missing = {'model': {'full_precision': {'a': 2, 'b': 3}}, 'optimizer': \
        {'a': {}, 'shared_optimizer_state': {'step': np.arange(5)}}}
    optimizer_state_key_unexpected = {'model': {'full_precision': {'a': 2, 'b': 3}}, 'optimizer': \
        {'a': {'Moment_1': np.arange(5), 'Moment_2': np.arange(7)}, 'shared_optimizer_state': {'step': np.arange(5), 'another_step': np.arange(1)}}}

    input_arguments = [
        (training_session_state_dict, precision_key_missing, ['full_precision']),
        (training_session_state_dict, precision_key_unexpected, ['mixed_precision']),
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
        'full_precision': {
            'a': np.arange(5),
            'b': np.arange(7)
        }
    }
    training_session_mock = _training_session_mock(model_states, {}, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict()
    assert (state_dict['model']['full_precision']['a'] == np.arange(5)).all()
    assert (state_dict['model']['full_precision']['b'] == np.arange(7)).all()

@patch('onnx.ModelProto')
def test_training_session_provides_model_states_pytorch_format(onnx_model_mock):
    trainer = _create_trainer()
    model_states = {
        'full_precision': {
            'a': np.arange(5),
            'b': np.arange(7)
        }
    }
    training_session_mock = _training_session_mock(model_states, {}, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict(pytorch_format=True)
    assert torch.all(torch.eq(state_dict['a'], torch.tensor(np.arange(5))))
    assert torch.all(torch.eq(state_dict['b'], torch.tensor(np.arange(7))))

@patch('onnx.ModelProto')
def test_onnx_graph_provides_frozen_model_states(onnx_model_mock):
    trainer = _create_trainer()
    model_states = {
        'full_precision': {
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
    assert (state_dict['model']['full_precision']['a'] == np.arange(5)).all()
    assert (state_dict['model']['full_precision']['b'] == np.arange(7)).all()
    assert (state_dict['model']['full_precision']['a_frozen_weight'] == np.array([1, 2, 3], dtype=np.float32)).all()
    assert 'a_non_fronzen_weight' not in state_dict['model']['full_precision']
    assert (state_dict['model']['full_precision']['a_float16_weight'] == np.array([7, 8, 9], dtype=np.float32)).all()

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
    assert (state_dict['optimizer']['model_weight']['Moment_1'] == np.arange(5)).all()
    assert (state_dict['optimizer']['model_weight']['Moment_2'] == np.arange(7)).all()
    assert (state_dict['optimizer']['shared_optimizer_state']['step'] == np.arange(1)).all()

@patch('onnx.ModelProto')
def test_training_session_provides_optimizer_states_pytorch_format(onnx_model_mock):
    trainer = _create_trainer()
    model_states = {
        'full_precision': {
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
    training_session_mock = _training_session_mock(model_states, optimizer_states, {})
    trainer._training_session = training_session_mock
    trainer._onnx_model = onnx_model_mock()

    state_dict = trainer.state_dict(pytorch_format=True)
    assert 'optimizer' not in state_dict

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
        'full_precision': {
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
    assert (state_dict['model']['full_precision']['a'] == np.arange(5)).all()
    assert (state_dict['model']['full_precision']['b'] == np.arange(7)).all()
    assert (state_dict['optimizer']['model_weight']['Moment_1'] == np.arange(5)).all()
    assert (state_dict['optimizer']['model_weight']['Moment_2'] == np.arange(7)).all()
    assert (state_dict['optimizer']['shared_optimizer_state']['step'] == np.arange(1)).all()
    assert state_dict['partition_info']['a']['original_dim'] == [1, 2, 3]

def test_load_state_dict_holds_when_training_session_not_initialized():
    trainer = _create_trainer()
    state_dict = {
        'model': {
            'full_precision': {
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

@pytest.mark.parametrize("state_dict, input_state_dict, error_key", [
    ({
        'optimizer':{},
    },
    {
        'optimizer':{},
        'trainer_options': {
            'optimizer_name': 'LambOptimizer'
        }
    },
    'model'),
    ({
        'model':{}
    },
    {
        'model':{},
        'trainer_options': {
            'optimizer_name': 'LambOptimizer'
        }
    },
    'optimizer')])
def test_load_state_dict_warns_when_model_optimizer_key_missing(state_dict, input_state_dict, error_key):
    trainer = _create_trainer()
    trainer._training_session = _training_session_mock({}, {}, {})
    trainer.state_dict = Mock(return_value=state_dict)
    trainer._update_onnx_model_initializers = Mock()
    trainer._init_session = Mock()
    with patch('onnx.ModelProto') as onnx_model_mock:
        trainer._onnx_model = onnx_model_mock()
        trainer._onnx_model.graph.initializer = []
        with pytest.warns(UserWarning) as user_warning:
            trainer.load_state_dict(input_state_dict)

    assert user_warning[0].message.args[0] == "Missing key: {} in state_dict".format(error_key)

@pytest.mark.parametrize("state_dict, input_state_dict, error_keys", _get_load_state_dict_strict_error_arguments())
def test_load_state_dict_errors_when_state_dict_mismatch(state_dict, input_state_dict, error_keys):
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
            'full_precision': {
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
            'full_precision': {
                'a': np.array([1, 2]),
                'b': np.array([3, 4])
            }
        },
        'optimizer': {
            'a': {
                'Moment_1': np.array([5, 6]),
                'Moment_2': np.array([7, 8])
            },
            'shared_optimizer_state': {
                'step': np.array([9])
            }
        },
        'trainer_options': {
            'optimizer_name': 'LambOptimizer'
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

    assert (state_dict_to_load[0]['a']['Moment_1'] ==  np.array([5, 6])).all()
    assert (state_dict_to_load[0]['a']['Moment_2'] ==  np.array([7, 8])).all()
    assert (state_dict_to_load[0]['shared_optimizer_state']['step'] ==  np.array([9])).all()

@patch('onnxruntime.training._checkpoint_storage.save')
def test_save_checkpoint_calls_checkpoint_storage_save(save_mock):
    trainer = _create_trainer()
    state_dict = {
        'model': {},
        'optimizer': {}
    }
    trainer.state_dict = Mock(return_value=state_dict)

    trainer.save_checkpoint('abc')

    save_args, _ = save_mock.call_args
    assert 'model' in save_args[0]
    assert not bool(save_args[0]['model'])
    assert 'optimizer' in save_args[0]
    assert not bool(save_args[0]['optimizer'])
    assert save_args[1] == 'abc'

@patch('onnxruntime.training._checkpoint_storage.save')
def test_save_checkpoint_exclude_optimizer_states(save_mock):
    trainer = _create_trainer()
    state_dict = {
        'model': {},
        'optimizer': {}
    }
    trainer.state_dict = Mock(return_value=state_dict)

    trainer.save_checkpoint('abc', include_optimizer_states=False)

    save_args, _ = save_mock.call_args
    assert 'model' in save_args[0]
    assert not bool(save_args[0]['model'])
    assert 'optimizer' not in save_args[0]
    assert save_args[1] == 'abc'

@patch('onnxruntime.training._checkpoint_storage.save')
def test_save_checkpoint_user_dict(save_mock):
    trainer = _create_trainer()
    state_dict = {
        'model': {},
        'optimizer': {}
    }
    trainer.state_dict = Mock(return_value=state_dict)

    trainer.save_checkpoint('abc', user_dict={'abc': np.arange(4)})

    save_args, _ = save_mock.call_args
    assert 'user_dict' in save_args[0]
    assert save_args[0]['user_dict'] == _checkpoint_storage.to_serialized_hex({'abc': np.arange(4)})

@patch('onnxruntime.training._checkpoint_storage.load')
@patch('onnxruntime.training.checkpoint.aggregate_checkpoints')
def test_load_checkpoint(aggregate_checkpoints_mock, load_mock):
    trainer = _create_trainer()
    trainer_options = {
        'mixed_precision': np.bool_(False),
        'world_rank': np.int64(0),
        'world_size': np.int64(1),
        'horizontal_parallel_size' : np.int64(1),
        'data_parallel_size' : np.int64(1),
        'zero_stage': np.int64(0)
    }
    state_dict = {
        'model': {},
        'optimizer': {},
        'trainer_options': {
            'mixed_precision': np.bool_(False),
            'world_rank': np.int64(0),
            'world_size': np.int64(1),
            'horizontal_parallel_size' : np.int64(1),
            'data_parallel_size' : np.int64(1),
            'zero_stage': np.int64(0)
        }
    }
    trainer.load_state_dict = Mock()

    load_mock.side_effect = [trainer_options, state_dict]
    trainer.load_checkpoint('abc')

    args_list = load_mock.call_args_list
    load_args, load_kwargs = args_list[0]
    assert load_args[0] == 'abc'
    assert load_kwargs['key'] == 'trainer_options'
    load_args, load_kwargs = args_list[1]
    assert load_args[0] == 'abc'
    assert 'key' not in load_kwargs
    assert not aggregate_checkpoints_mock.called

@patch('onnxruntime.training._checkpoint_storage.load')
@patch('onnxruntime.training.checkpoint.aggregate_checkpoints')
@pytest.mark.parametrize("trainer_options", [
    {
        'mixed_precision': np.bool_(False),
        'world_rank': np.int64(0),
        'world_size': np.int64(4),
        'horizontal_parallel_size' : np.int64(1),
        'data_parallel_size' : np.int64(4),
        'zero_stage': np.int64(1)
    },
    {
        'mixed_precision': np.bool_(True),
        'world_rank': np.int64(0),
        'world_size': np.int64(1),
        'horizontal_parallel_size' : np.int64(1),
        'data_parallel_size' : np.int64(1),
        'zero_stage': np.int64(1)
    },
    {
        'mixed_precision': np.bool_(True),
        'world_rank': np.int64(0),
        'world_size': np.int64(1),
        'horizontal_parallel_size' : np.int64(1),
        'data_parallel_size' : np.int64(1),
        'zero_stage': np.int64(1)
    }
])
def test_load_checkpoint_aggregation_required_zero_enabled(aggregate_checkpoints_mock, load_mock, trainer_options):
    trainer = _create_trainer()
    trainer.load_state_dict = Mock()

    load_mock.side_effect = [trainer_options]
    trainer.load_checkpoint('abc')

    args_list = load_mock.call_args_list
    load_args, load_kwargs = args_list[0]
    assert load_args[0] == 'abc'
    assert load_kwargs['key'] == 'trainer_options'
    assert aggregate_checkpoints_mock.called
    call_args, _ = aggregate_checkpoints_mock.call_args
    assert call_args[0] == tuple(['abc'])

@patch('onnxruntime.training._checkpoint_storage.load')
@patch('onnxruntime.training.checkpoint.aggregate_checkpoints')
def test_load_checkpoint_user_dict(aggregate_checkpoints_mock, load_mock):
    trainer = _create_trainer()
    trainer_options = {
        'mixed_precision': np.bool_(False),
        'world_rank': np.int64(0),
        'world_size': np.int64(1),
        'horizontal_parallel_size': np.int64(1),
        'data_parallel_size': np.int64(1),
        'zero_stage': np.int64(0)
    }
    state_dict = {
        'model': {},
        'optimizer': {},
        'trainer_options': {
            'mixed_precision': np.bool_(False),
            'world_rank': np.int64(0),
            'world_size': np.int64(1),
            'horizontal_parallel_size': np.int64(1),
            'data_parallel_size': np.int64(1),
            'zero_stage': np.int64(0)
        },
        'user_dict': _checkpoint_storage.to_serialized_hex({'array': torch.tensor(np.arange(5))})
    }
    trainer.load_state_dict = Mock()

    load_mock.side_effect = [trainer_options, state_dict]
    user_dict = trainer.load_checkpoint('abc')

    assert torch.all(torch.eq(user_dict['array'], torch.tensor(np.arange(5))))

@patch('onnxruntime.training._checkpoint_storage.load')
def test_checkpoint_aggregation(load_mock):
    trainer_options1 = {
        'mixed_precision': np.bool_(False),
        'world_rank': np.int64(0),
        'world_size': np.int64(2),
        'horizontal_parallel_size' : np.int64(1),
        'data_parallel_size' : np.int64(2),
        'zero_stage': np.int64(1),
        'optimizer_name': b'Adam'
    }
    trainer_options2 = {
        'mixed_precision': np.bool_(False),
        'world_rank': np.int64(1),
        'world_size': np.int64(2),
        'horizontal_parallel_size' : np.int64(1),
        'data_parallel_size' : np.int64(2),
        'zero_stage': np.int64(1),
        'optimizer_name': b'Adam'
    }

    state_dict1 = {
        'model': {
            'full_precision': {
                'optimizer_sharded': np.array([1, 2, 3]),
                'non_sharded': np.array([11, 22, 33])
            }
        },
        'optimizer': {
            'optimizer_sharded': {
                'Moment_1': np.array([9, 8, 7]),
                'Moment_2': np.array([99, 88, 77]),
                'Step': np.array([5])
            },
            'non_sharded': {
                'Moment_1': np.array([666, 555, 444]),
                'Moment_2': np.array([6666, 5555, 4444]),
                'Step': np.array([55])
            }
        },
        'trainer_options': {
            'mixed_precision': np.bool_(False),
            'world_rank': np.int64(0),
            'world_size': np.int64(1),
            'horizontal_parallel_size' : np.int64(1),
            'data_parallel_size' : np.int64(1),
            'zero_stage': np.int64(0),
            'optimizer_name': b'Adam'
        },
        'partition_info': {
            'optimizer_sharded': {'original_dim': np.array([2, 3])}
        }
    }

    state_dict2 = {
        'model': {
            'full_precision': {
                'optimizer_sharded': np.array([1, 2, 3]),
                'non_sharded': np.array([11, 22, 33])
            }
        },
        'optimizer': {
            'optimizer_sharded': {
                'Moment_1': np.array([6, 5, 4]),
                'Moment_2': np.array([66, 55, 44]),
                'Step': np.array([5])
            },
            'non_sharded': {
                'Moment_1': np.array([666, 555, 444]),
                'Moment_2': np.array([6666, 5555, 4444]),
                'Step': np.array([55])
            }
        },
        'trainer_options': {
            'mixed_precision': np.bool_(False),
            'world_rank': np.int64(1),
            'world_size': np.int64(1),
            'horizontal_parallel_size' : np.int64(1),
            'data_parallel_size' : np.int64(1),
            'zero_stage': np.int64(0),
            'optimizer_name': b'Adam'
        },
        'partition_info': {
            'optimizer_sharded': {'original_dim': np.array([2, 3])}
        }
    }

    load_mock.side_effect = [trainer_options1, trainer_options2, trainer_options1, state_dict1, state_dict2]
    state_dict = checkpoint.aggregate_checkpoints(['abc', 'def'], pytorch_format=False)

    assert (state_dict['model']['full_precision']['optimizer_sharded'] == np.array([1, 2, 3])).all()
    assert (state_dict['model']['full_precision']['non_sharded'] == np.array([11, 22, 33])).all()
    assert (state_dict['optimizer']['optimizer_sharded']['Moment_1'] == np.array([[9, 8, 7], [6, 5, 4]])).all()
    assert (state_dict['optimizer']['optimizer_sharded']['Moment_2'] == np.array([[99, 88, 77], [66, 55, 44]])).all()
    assert (state_dict['optimizer']['optimizer_sharded']['Step'] == np.array([5])).all()
    assert (state_dict['optimizer']['non_sharded']['Moment_1'] == np.array([666, 555, 444])).all()
    assert (state_dict['optimizer']['non_sharded']['Moment_2'] == np.array([6666, 5555, 4444])).all()
    assert (state_dict['optimizer']['non_sharded']['Step'] == np.array([55])).all()

    assert state_dict['trainer_options']['mixed_precision'] == False
    assert state_dict['trainer_options']['world_rank'] == 0
    assert state_dict['trainer_options']['world_size'] == 1
    assert state_dict['trainer_options']['horizontal_parallel_size'] == 1
    assert state_dict['trainer_options']['data_parallel_size'] == 1
    assert state_dict['trainer_options']['zero_stage'] == 0
    assert state_dict['trainer_options']['optimizer_name'] == b'Adam'

@patch('onnxruntime.training._checkpoint_storage.load')
def test_checkpoint_aggregation_mixed_precision(load_mock):
    trainer_options1 = {
        'mixed_precision': np.bool_(True),
        'world_rank': np.int64(0),
        'world_size': np.int64(2),
        'horizontal_parallel_size': np.int64(1),
        'data_parallel_size': np.int64(2),
        'zero_stage': np.int64(1),
        'optimizer_name': b'Adam'
    }
    trainer_options2 = {
        'mixed_precision': np.bool_(True),
        'world_rank': np.int64(1),
        'world_size': np.int64(2),
        'horizontal_parallel_size': np.int64(1),
        'data_parallel_size': np.int64(2),
        'zero_stage': np.int64(1),
        'optimizer_name': b'Adam'
    }

    state_dict1 = {
        'model': {
            'full_precision': {
                'sharded': np.array([1, 2, 3]),
                'non_sharded': np.array([11, 22, 33])
            }
        },
        'optimizer': {
            'sharded': {
                'Moment_1': np.array([9, 8, 7]),
                'Moment_2': np.array([99, 88, 77]),
                'Step': np.array([5])
            },
            'non_sharded': {
                'Moment_1': np.array([666, 555, 444]),
                'Moment_2': np.array([6666, 5555, 4444]),
                'Step': np.array([55])
            }
        },
        'trainer_options': {
            'mixed_precision': np.bool_(True),
            'world_rank': np.int64(0),
            'world_size': np.int64(1),
            'horizontal_parallel_size': np.int64(1),
            'data_parallel_size': np.int64(1),
            'zero_stage': np.int64(0),
            'optimizer_name': b'Adam'
        },
        'partition_info': {
            'sharded': {'original_dim': np.array([2, 3])}
        }
    }

    state_dict2 = {
        'model': {
            'full_precision': {
                'sharded': np.array([4, 5, 6]),
                'non_sharded': np.array([11, 22, 33])
            }
        },
        'optimizer': {
            'sharded': {
                'Moment_1': np.array([6, 5, 4]),
                'Moment_2': np.array([66, 55, 44]),
                'Step': np.array([5])
            },
            'non_sharded': {
                'Moment_1': np.array([666, 555, 444]),
                'Moment_2': np.array([6666, 5555, 4444]),
                'Step': np.array([55])
            }
        },
        'trainer_options': {
            'mixed_precision': np.bool_(True),
            'world_rank': np.int64(1),
            'world_size': np.int64(1),
            'horizontal_parallel_size': np.int64(1),
            'data_parallel_size': np.int64(1),
            'zero_stage': np.int64(0),
            'optimizer_name': b'Adam'
        },
        'partition_info': {
            'sharded': {'original_dim': np.array([2, 3])}
        }
    }

    load_mock.side_effect = [trainer_options1, trainer_options2, trainer_options1, state_dict1, state_dict2]
    state_dict = checkpoint.aggregate_checkpoints(['abc', 'def'], pytorch_format=False)

    assert (state_dict['model']['full_precision']['sharded'] == np.array([[1, 2, 3], [4, 5, 6]])).all()
    assert (state_dict['model']['full_precision']['non_sharded'] == np.array([11, 22, 33])).all()
    assert (state_dict['optimizer']['sharded']['Moment_1'] == np.array([[9, 8, 7], [6, 5, 4]])).all()
    assert (state_dict['optimizer']['sharded']['Moment_2'] == np.array([[99, 88, 77], [66, 55, 44]])).all()
    assert (state_dict['optimizer']['sharded']['Step'] == np.array([5])).all()
    assert (state_dict['optimizer']['non_sharded']['Moment_1'] == np.array([666, 555, 444])).all()
    assert (state_dict['optimizer']['non_sharded']['Moment_2'] == np.array([6666, 5555, 4444])).all()
    assert (state_dict['optimizer']['non_sharded']['Step'] == np.array([55])).all()

    assert state_dict['trainer_options']['mixed_precision'] == True
    assert state_dict['trainer_options']['world_rank'] == 0
    assert state_dict['trainer_options']['world_size'] == 1
    assert state_dict['trainer_options']['horizontal_parallel_size'] == 1
    assert state_dict['trainer_options']['data_parallel_size'] == 1
    assert state_dict['trainer_options']['zero_stage'] == 0
    assert state_dict['trainer_options']['optimizer_name'] == b'Adam'
