import cerberus
import torch

from .optim import lr_scheduler
from .amp import loss_scaler


class PytorchTrainerOptions(object):
    r"""Settings used by ONNX Runtime training backend

    The parameters are hierarchically organized to facilitate configuration through semantic groups
    that encompasses features, such as distributed training, etc.

    Although this class uses a dictionary for initialization, input validation
    is performed to make sure only supported parameters and values are stored.

    Args:
        options (dict): contains all training options
        _validate (bool, default is True) : internal flag that indicates whether input should be validated

    Supported schema for kwargs:

    .. code-block:: python

    schema = {
                'batch' : {
                    'type' : 'dict',
                    'schema' : {
                        'gradient_accumulation_steps' : {
                            'type' : 'integer',
                            'min' : 0,
                            'default' : 0
                        }
                    },
                },
                'device' : {
                    'type' : 'dict',
                    'schema' : {
                        'id' : {
                            'type' : 'string',
                            'nullable' : True,
                            'default' : None
                        },
                        'mem_limit' : {
                            'type' : 'integer',
                            'min' : 0,
                            'default' : 0
                        }
                    }
                },
                'distributed' : {
                    'type' : 'dict',
                    'schema' : {
                        'world_rank' : {
                            'type' : 'integer',
                            'min' : 0,
                            'default' : 0
                        },
                        'world_size' : {
                            'type' : 'integer',
                            'min' : 1,
                            'default' : 1
                        },
                        'local_rank' : {
                            'type' : 'integer',
                            'min' : 0,
                            'default' : 0
                        },
                        'allreduce_post_accumulation' : {
                            'type' : 'boolean',
                            'default' : False
                        },
                        'enable_partition_optimizer' : {
                            'type' : 'boolean',
                            'default' : False
                        },
                        'enable_adasum' : {
                            'type' : 'boolean',
                            'default' : False
                        }
                    }
                },
                'lr_scheduler' : {
                    'type' : 'optim.lr_scheduler',
                    'nullable' : True,
                    'default' : None
                },
                'mixed_precision' : {
                    'type' : 'dict',
                    'schema' : {
                        'enabled' : {
                            'type' : 'boolean',
                            'default' : False
                        },
                        'loss_scaler' : {
                            'type' : 'amp.loss_scaler',
                            'nullable' : True,
                            'default' : None
                        }
                    }
                },
                'utils' : {
                    'type' : 'dict',
                    'schema' : {
                        'grad_norm_clip' : {
                            'type' : 'boolean',
                            'default' : False
                        }
                    }
                },
                '_internal_use' : {
                    'type' : 'dict',
                    'schema' : {
                        'frozen_weights' : {
                            'type' : 'list',
                            'default' : []
                        },
                        'enable_internal_postprocess' : {
                            'type' : 'boolean',
                            'default' : True
                        },
                        'extra_postprocess' : {
                            'check_with' : 'callable',
                            'nullable' : True,
                            'default' : None
                        }
                    }
                }
             }

    Keyword arguments:
        batch (dict):
            batch related settings
        batch.gradient_accumulation_steps (int, 0):
            number of steps to accumulate before do collective gradient reduction
        device (dict):
            compute device related settings
        device.id (string, default is None):
            device to run training
        device.mem_limit (int):
            maximum memory size (in bytes) used by device.id
        distributed (dict):
            distributed training options
        distributed.world_rank (int, default is 0):
            rank ID used for data parallelism
        distributed.world_size (int, default is 1):
            number of rank participating in data parallelism
        distributed.allreduce_post_accumulation (bool, default is False):
            True enables overlap of AllReduce with computation, while False,
            postpone AllReduce until all gradients are ready
        distributed.enable_partition_optimizer (bool, default is False):
            enable or disable partition of optimizer state (ZeRO algorithm)
        distributed.enable_adasum (bool, default is False):
            enable `Adasum <https://github.com/horovod/horovod/pull/1484>`_
            algorithm for AllReduce
        lr_scheduler (optim.LRScheduler, default is None):
            specifies learning rate scheduler
        mixed_precision (dict):
            mixed precision training options
        mixed_precision.enabled (bool, default is False):
            enable mixed precision (fp16)
        mixed_precision.loss_scaler (amp.LossScaler, default is None):
            specifies a loss scaler to be used for fp16. If not specified,
            :py:class:`.DynamicLossScaler` is used with default values.
            Users can also instantiate :py:class:`.DynamicLossScaler` and
            override its parameters. Lastly, a completely new implementation
            can be specified by extending :py:class:`.LossScaler` class from scratch
        utils (dict):
            miscellaneous options
        utils.grad_norm_clip (bool, default is False):
            enables gradient norm clipping for 'AdamOptimizer' and 'LambOptimizer'
        _internal_use (dict):
            internal, possibly undocumented, options that might be removed in the next release
        _internal_use.frozen_weights (list, []):
            list of model parameters to freeze (stop training)
        _internal_use.enable_internal_postprocess (bool, default is True):
            enable internal internal post processing of the ONNX model
        _internal_use.extra_postprocess (callable, default is None)
            a functor to postprocess the ONNX model.
            It does not override :py:attr:`._internal_use.enable_internal_postprocess`, but complement it

    Example:
        .. code-block:: python

            opts = PytorchTrainerOptions({
                               'batch' : {
                                   'gradient_accumulation_steps' : 128
                               },
                               'device' : {
                                   'id' : 'cuda:0',
                                   'mem_limit' : 2*1024*1024*1024,
                               },
                               'lr_scheduler' : optim.lr_scheduler.LinearWarmupLRScheduler(),
                               'mixed_precision' : {
                                   'enabled': True,
                                   'loss_scaler': amp.LossScaler(loss_scale=float(1 << 16))
                               }
            })
            fp16_enabled = opts.mixed_precision.enabled
     """

    def __init__(self, options, _validate=True):
        # Keep a copy of original input for debug
        self._original_opts = dict(options)

        # Add an empty dictionary for non specified nested dicts
        subgroups = [k for k, v in _PYTORCH_TRAINER_OPTIONS_SCHEMA.items()
                     if isinstance(v, dict) and 'type' in v and v['type'] == 'dict']
        self._validated_opts = dict(self._original_opts)
        if _validate:
            for name in subgroups:
                if name not in self._validated_opts:
                    self._validated_opts[name] = {}

            # Validates user input
            validator = PytorchTrainerOptionsValidator(_PYTORCH_TRAINER_OPTIONS_SCHEMA)
            self._validated_opts = validator.validated(self._validated_opts)
            if self._validated_opts is None:
                self._validated_opts = validator.errors
            _validate = False

        # Convert dict in object
        for k, v in self._validated_opts.items():
            setattr(self, k, self._wrap(v, _validate))

        # Keep this in the last line
        self._initialized = True

    def __repr__(self):
        return '{%s}' % str(', '.join("'%s': %s" % (k, repr(v))
                                      for (k, v) in self.__dict__.items()
                                      if k not in ['_original_opts', '_validated_opts', '_initialized']))

    def __setattr__(self, k, v):
        if hasattr(self, '_initialized'):
            raise Exception(f"{self.__class__.__name__} is an immutable class")
        return super().__setattr__(k, v)

    def _wrap(self, v, validate):
        if isinstance(v, (tuple, list, set, frozenset)):
            return type(v)([self._wrap(v, validate) for v in v])
        else:
            return PytorchTrainerOptions(v, False) if isinstance(v, dict) else v


class PytorchTrainerOptionsValidator(cerberus.Validator):
    _LR_SCHEDULER = cerberus.TypeDefinition(
        'lr_scheduler', (lr_scheduler.LRScheduler,), ())
    _LOSS_SCALER = cerberus.TypeDefinition(
        'loss_scaler', (loss_scaler.LossScaler,), ())

    types_mapping = cerberus.Validator.types_mapping.copy()
    types_mapping['lr_scheduler'] = _LR_SCHEDULER
    types_mapping['loss_scaler'] = _LOSS_SCALER


def _check_is_callable(field, value, error):
    result = False
    try:
        # Python 3
        result = value is None or callable(value)
    except:
        # Python 3 but < 3.2
        if hasattr(value, '__call__'):
            result = True
    if not result:
        error(field, "Must be callable or None")


_PYTORCH_TRAINER_OPTIONS_SCHEMA = {
    'batch': {
        'type': 'dict',
        'schema': {
            'gradient_accumulation_steps': {
                'type': 'integer',
                'min': 0,
                'default': 0
            }
        },
    },
    'device': {
        'type': 'dict',
        'schema': {
            'id': {
                'type': 'string',
                'nullable': True,
                'default': None
            },
            'mem_limit': {
                'type': 'integer',
                'min': 0,
                'default': 0
            }
        }
    },
    'distributed': {
        'type': 'dict',
        'schema': {
            'world_rank': {
                'type': 'integer',
                'min': 0,
                'default': 0
            },
            'world_size': {
                'type': 'integer',
                'min': 1,
                'default': 1
            },
            'local_rank': {
                'type': 'integer',
                'min': 0,
                'default': 0
            },
            'allreduce_post_accumulation': {
                'type': 'boolean',
                'default': False
            },
            'enable_partition_optimizer': {
                'type': 'boolean',
                'default': False
            },
            'enable_adasum': {
                'type': 'boolean',
                'default': False
            }

        }
    },
    'lr_scheduler': {
        'type': 'lr_scheduler',
        'nullable': True,
        'default': None
    },
    'mixed_precision': {
        'type': 'dict',
        'schema': {
            'enabled': {
                'type': 'boolean',
                'default': False
            },
            'loss_scaler': {
                'type': 'loss_scaler',
                'nullable': True,
                'default': None
            }
        }
    },
    'utils': {
        'type': 'dict',
        'schema': {
            'grad_norm_clip': {
                'type': 'boolean',
                'default': False
            }
        }
    },
    '_internal_use': {
        'type': 'dict',
        'schema': {
            'frozen_weights': {
                'type': 'list',
                'default': []
            },
            'enable_internal_postprocess': {
                'type': 'boolean',
                'default': True
            },
            'extra_postprocess': {
                'check_with': _check_is_callable,
                'nullable': True,
                'default': None

            }
        }
    }
}
