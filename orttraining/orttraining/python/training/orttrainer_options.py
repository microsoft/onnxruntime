import cerberus
import torch

from .optim import lr_scheduler
from .amp import loss_scaler
from . import PropagateCastOpsStrategy
import onnxruntime as ort

class ORTTrainerOptions(object):
    r"""Settings used by ONNX Runtime training backend

    The parameters are hierarchically organized to facilitate configuration through semantic groups
    that encompasses features, such as distributed training, etc.

    Input validation is performed on the input dict during instantiation to ensure
    that supported parameters and values are passed in. Invalid input results
    in :py:obj:`ValueError` exception with details on it.

    Args:
        options (dict): contains all training options
        _validate (bool, default is True): for internal use only

    Supported schema for kwargs:

    .. code-block:: python

    schema = {
                'batch' : {
                    'type' : 'dict',
                    'required': False,
                    'default' : {},
                    'schema' : {
                        'gradient_accumulation_steps' : {
                            'type' : 'integer',
                            'min' : 1,
                            'default' : 1
                        }
                    },
                },
                'device' : {
                    'type' : 'dict',
                    'required': False,
                    'default' : {},
                    'schema' : {
                        'id' : {
                            'type' : 'string',
                            'default' : 'cuda'
                        },
                        'mem_limit' : {
                            'type' : 'integer',
                            'min' : 0,
                            'default' : 0
                        }
                    }
                },
                'distributed': {
                    'type': 'dict',
                    'default': {},
                    'required': False,
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
                        'data_parallel_size': {
                            'type': 'integer',
                            'min': 1,
                            'default': 1
                        },
                        'horizontal_parallel_size': {
                            'type': 'integer',
                            'min': 1,
                            'default': 1
                        },
                        'pipeline_parallel' : {
                            'type': 'dict',
                            'default': {},
                            'required': False,
                            'schema': {
                                'pipeline_parallel_size': {
                                    'type': 'integer',
                                    'min': 1,
                                    'default': 1
                                },
                                'num_pipeline_micro_batches': {
                                    'type': 'integer',
                                    'min': 1,
                                    'default': 1
                                },
                                'pipeline_cut_info_string': {
                                    'type': 'string',
                                    'default': ''
                                },
                                'sliced_schema': {
                                    'type': 'dict',
                                    'default': {},
                                    'keysrules': {'type': 'string'},
                                    'valuesrules': {
                                        'type': 'list',
                                        'schema': {'type': 'integer'}
                                    }
                                },
                                'sliced_axes': {
                                    'type': 'dict',
                                    'default': {},
                                    'keysrules': {'type': 'string'},
                                    'valuesrules': {'type': 'integer'}
                                },
                                'sliced_tensor_names': {
                                    'type': 'list',
                                    'schema': {'type': 'string'},
                                    'default': []
                                }
                            }
                        },
                        'allreduce_post_accumulation': {
                            'type': 'boolean',
                            'default': False
                        },
                        'deepspeed_zero_optimization': {
                            'type': 'dict',
                            'default': {},
                            'required': False,
                            'schema': {
                                'stage': {
                                    'type': 'integer',
                                    'min': 0,
                                    'max': 1,
                                    'default': 0
                                },
                            }
                        },
                        'enable_adasum': {
                            'type': 'boolean',
                            'default': False
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
                    'required': False,
                    'default' : {},
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
                'graph_transformer': {
                    'type': 'dict',
                    'required': False,
                    'default': {},
                    'schema': {
                        'attn_dropout_recompute': {
                            'type': 'boolean',
                            'default': False
                        },
                        'gelu_recompute': {
                            'type': 'boolean',
                            'default': False
                        },
                        'transformer_layer_recompute': {
                            'type': 'boolean',
                            'default': False
                        },
                        'number_recompute_layers': {
                            'type': 'integer',
                            'min': 0,
                            'default': 0
                        },
                        'propagate_cast_ops_config': {
                            'type': 'dict',
                            'required': False,
                            'default': {},
                            'schema': {
                                'propagate_cast_ops_strategy': {
                                    'type': 'onnxruntime.training.PropagateCastOpsStrategy',
                                    'default': INSERT_AND_REDUCE
                                },
                                'propagate_cast_ops_level': {
                                    'type': 'integer',
                                    'default': -1
                                },
                                'propagate_cast_ops_allow': {
                                    'type': 'list',
                                    'schema': {'type': 'string'},
                                    'default': []
                                }
                            }
                        },
                        'allow_layer_norm_mod_precision': {
                            'type': 'boolean',
                            'default': False
                        }
                    }
                },
                'utils' : {
                    'type' : 'dict',
                    'required': False,
                    'default' : {},
                    'schema' : {
                        'frozen_weights' : {
                            'type' : 'list',
                            'default' : []
                        },
                        'grad_norm_clip' : {
                            'type' : 'boolean',
                            'default' : True
                        },
                        'invertible_layer_norm_gradient' : {
                            'type' : 'boolean',
                            'default' : False
                        },
                        'run_symbolic_shape_infer' : {
                            'type' : 'boolean',
                            'default' : False
                        }
                    }
                },
                'debug' : {
                    'type' : 'dict',
                    'required': False,
                    'default' : {},
                    'schema' : {
                        'deterministic_compute' : {
                            'type' : 'boolean',
                            'default' : False
                        },
                        'check_model_export' : {
                            'type' : 'boolean',
                            'default' : False
                        },
                        'graph_save_paths' : {
                            'type' : 'dict',
                            'default': {},
                            'required': False,
                            'schema': {
                                'model_after_graph_transforms_path': {
                                    'type': 'string',
                                    'default': ''
                                },
                                'model_with_gradient_graph_path':{
                                    'type': 'string',
                                    'default': ''
                                },
                                'model_with_training_graph_path': {
                                    'type': 'string',
                                    'default': ''
                                },
                                'model_with_training_graph_after_optimization_path': {
                                    'type': 'string',
                                    'default': ''
                                },
                            }
                        },
                    }
                },
                '_internal_use' : {
                    'type' : 'dict',
                    'required': False,
                    'default' : {},
                    'schema' : {
                        'enable_internal_postprocess' : {
                            'type' : 'boolean',
                            'default' : True
                        },
                        'extra_postprocess' : {
                            'type' : 'callable',
                            'nullable' : True,
                            'default' : None
                        },
                        'onnx_opset_version': {
                            'type': 'integer',
                            'min' : 12,
                            'max' : 13,
                            'default': 12
                        },
                        'enable_onnx_contrib_ops' : {
                            'type' : 'boolean',
                            'default' : True
                        }
                    }
                },
                'provider_options':{
                    'type': 'dict',
                    'default': {},
                    'required': False,
                    'schema': {}
                },
                'session_options': {
                    'type': 'SessionOptions',
                    'nullable': True,
                    'default': None
                },
             }

    Keyword arguments:
        batch (dict):
            batch related settings
        batch.gradient_accumulation_steps (int, default is 1):
            number of steps to accumulate before do collective gradient reduction
        device (dict):
            compute device related settings
        device.id (string, default is 'cuda'):
            device to run training
        device.mem_limit (int):
            maximum memory size (in bytes) used by device.id
        distributed (dict):
            distributed training options.
        distributed.world_rank (int, default is 0):
            rank ID used for data/horizontal parallelism
        distributed.world_size (int, default is 1):
            number of ranks participating in parallelism
        distributed.data_parallel_size (int, default is 1):
            number of ranks participating in data parallelism
        distributed.horizontal_parallel_size (int, default is 1):
            number of ranks participating in horizontal parallelism
        distributed.pipeline_parallel (dict):
            Options which are only useful to pipeline parallel.
        distributed.pipeline_parallel.pipeline_parallel_size (int, default is 1):
            number of ranks participating in pipeline parallelism
        distributed.pipeline_parallel.num_pipeline_micro_batches (int, default is 1):
            number of micro-batches. We divide input batch into micro-batches and run the graph.
        distributed.pipeline_parallel.pipeline_cut_info_string (string, default is ''):
            string of cutting ids for pipeline partition.
        distributed.allreduce_post_accumulation (bool, default is False):
            True enables overlap of AllReduce with computation, while False,
            postpone AllReduce until all gradients are ready
        distributed.deepspeed_zero_optimization:
            DeepSpeed ZeRO options.
        distributed.deepspeed_zero_optimization.stage (int, default is 0):
            select which stage of DeepSpeed ZeRO to use. Stage 0 means disabled.
        distributed.enable_adasum (bool, default is False):
            enable `Adasum <https://arxiv.org/abs/2006.02924>`_
            algorithm for AllReduce
        lr_scheduler (optim._LRScheduler, default is None):
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
        graph_transformer (dict):
            graph transformer related configurations
        graph_transformer.attn_dropout_recompute(bool, default False)
        graph_transformer.gelu_recompute(bool, default False)
        graph_transformer.transformer_layer_recompute(bool, default False)
        graph_transformer.number_recompute_layers(bool, default False)
        graph_transformer.propagate_cast_ops_config (dict):
            graph_transformer.propagate_cast_ops_config.strategy(PropagateCastOpsStrategy, default INSERT_AND_REDUCE)
                Specify the choice of the cast propagation optimization strategy, either, INSERT_AND_REDUCE or FLOOD_FILL.
                INSERT_AND_REDUCE strategy inserts and reduces cast operations around the nodes with allowed opcodes.
                FLOOD_FILL strategy expands float16 regions in the graph using the allowed opcodes, and unlike
                INSERT_AND_REDUCE does not touch opcodes outside expanded float16 region.
            graph_transformer.propagate_cast_ops_config.level(integer, default -1)
                Optimize by moving Cast operations if propagate_cast_ops_level is non-negative.
                Use predetermined list of opcodes considered safe to move before/after cast operation
                if propagate_cast_ops_level is positive and use propagate_cast_ops_allow otherwise.
            graph_transformer.propagate_cast_ops_config.allow(list of str, [])
                List of opcodes to be considered safe to move before/after cast operation if propagate_cast_ops_level is zero.
        graph_transformer.allow_layer_norm_mod_precision(bool, default False)
            Enable LayerNormalization/SimplifiedLayerNormalization fusion 
            even if it requires modified compute precision
        attn_dropout_recompute (bool, default is False):
            enable recomputing attention dropout to save memory
        gelu_recompute (bool, default is False):
            enable recomputing Gelu activation output to save memory
        transformer_layer_recompute (bool, default is False):
            enable recomputing transformer layerwise to save memory
        number_recompute_layers (int, default is 0)
            number of layers to apply transformer_layer_recompute, by default system will
            apply recompute to all the layers, except for the last one
        utils (dict):
            miscellaneous options
        utils.frozen_weights (list of str, []):
            list of model parameter names to skip training (weights don't change)
        utils.grad_norm_clip (bool, default is True):
            enables gradient norm clipping for 'AdamOptimizer' and 'LambOptimizer'
        utils.invertible_layer_norm_gradient (bool, default is False):
            enables use of invertible layer norm gradients
        utils.run_symbolic_shape_infer (bool, default is False):
            runs symbolic shape inference on the model
        debug (dict):
            debug options
        debug.deterministic_compute (bool, default is False)
            forces compute to be deterministic accross runs
        debug.check_model_export (bool, default is False)
            compares PyTorch model outputs with ONNX model outputs in inference before the first
            train step to ensure successful model export
        debug.graph_save_paths (dict):
            paths used for dumping ONNX graphs for debugging purposes
        debug.graph_save_paths.model_after_graph_transforms_path (str, default is "")
            path to export the ONNX graph after training-related graph transforms have been applied.
            No output when it is empty.
        debug.graph_save_paths.model_with_gradient_graph_path (str, default is "")
            path to export the ONNX graph with the gradient graph added. No output when it is empty.
        debug.graph_save_paths.model_with_training_graph_path (str, default is "")
            path to export the training ONNX graph with forward, gradient and optimizer nodes.
            No output when it is empty.
        debug.graph_save_paths.model_with_training_graph_after_optimization_path (str, default is "")
            outputs the optimized training graph to the path if nonempty.
        _internal_use (dict):
            internal options, possibly undocumented, that might be removed without notice
        _internal_use.enable_internal_postprocess (bool, default is True):
            enable internal internal post processing of the ONNX model
        _internal_use.extra_postprocess (callable, default is None)
            a functor to postprocess the ONNX model and return a new ONNX model.
            It does not override :py:attr:`._internal_use.enable_internal_postprocess`, but complement it
        _internal_use.onnx_opset_version (int, default is 12):
            ONNX opset version used during model exporting.
        _internal_use.enable_onnx_contrib_ops (bool, default is True)
            enable PyTorch to export nodes as contrib ops in ONNX.
            This flag may be removed anytime in the future.
        session_options (onnxruntime.SessionOptions):
            The SessionOptions instance that TrainingSession will use.
        provider_options (dict): 
            The provider_options for customized execution providers. it is dict map from EP name to 
            a key-value pairs, like {'EP1' : {'key1' : 'val1'}, ....}

    Example:
        .. code-block:: python

            opts = ORTTrainerOptions({
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

    def __init__(self, options={}):
        # Keep a copy of original input for debug
        self._original_opts = dict(options)

        # Used for logging purposes
        self._main_class_name = self.__class__.__name__

        # Validates user input
        self._validated_opts = dict(self._original_opts)
        validator = ORTTrainerOptionsValidator(
            _ORTTRAINER_OPTIONS_SCHEMA)
        self._validated_opts = validator.validated(self._validated_opts)
        if self._validated_opts is None:
            raise ValueError(f'Invalid options: {validator.errors}')

        # Convert dict in object
        for k, v in self._validated_opts.items():
            setattr(self, k, self._wrap(v))

    def __repr__(self):
        return '{%s}' % str(', '.join("'%s': %s" % (k, repr(v))
                                      for (k, v) in self.__dict__.items()
                                      if k not in ['_original_opts', '_validated_opts', '_main_class_name']))

    def _wrap(self, v):
        if isinstance(v, (tuple, list, set, frozenset)):
            return type(v)([self._wrap(i) for i in v])
        else:
            return _ORTTrainerOptionsInternal(self._main_class_name, v) if isinstance(v, dict) else v


class _ORTTrainerOptionsInternal(ORTTrainerOptions):
    r"""Internal class used by ONNX Runtime training backend for input validation

    NOTE: Users MUST NOT use this class in any way!
    """

    def __init__(self, main_class_name, options):
        # Used for logging purposes
        self._main_class_name = main_class_name
        # We don't call super().__init__(options) here but still called it "_validated_opts"
        # instead of "_original_opts" because it has been validated in the top-level
        # ORTTrainerOptions's constructor.
        self._validated_opts = dict(options)
        # Convert dict in object
        for k, v in dict(options).items():
            setattr(self, k, self._wrap(v))


class ORTTrainerOptionsValidator(cerberus.Validator):
    _LR_SCHEDULER = cerberus.TypeDefinition(
        'lr_scheduler', (lr_scheduler._LRScheduler,), ())
    _LOSS_SCALER = cerberus.TypeDefinition(
        'loss_scaler', (loss_scaler.LossScaler,), ())

    _SESSION_OPTIONS = cerberus.TypeDefinition(
        'session_options', (ort.SessionOptions,),())

    _PROPAGATE_CAST_OPS_STRATEGY = cerberus.TypeDefinition(
        "propagate_cast_ops_strategy", (PropagateCastOpsStrategy,),())

    types_mapping = cerberus.Validator.types_mapping.copy()
    types_mapping['lr_scheduler'] = _LR_SCHEDULER
    types_mapping['loss_scaler'] = _LOSS_SCALER
    types_mapping['session_options'] = _SESSION_OPTIONS
    types_mapping['propagate_cast_ops_strategy'] = _PROPAGATE_CAST_OPS_STRATEGY


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


_ORTTRAINER_OPTIONS_SCHEMA = {
    'batch': {
        'type': 'dict',
        'default_setter': lambda _: {},
        'required': False,
        'schema': {
            'gradient_accumulation_steps': {
                'type': 'integer',
                'min': 1,
                'default': 1
            }
        },
    },
    'device': {
        'type': 'dict',
        'default_setter': lambda _: {},
        'required': False,
        'schema': {
            'id': {
                'type': 'string',
                'default': 'cuda'
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
        'default_setter': lambda _: {},
        'required': False,
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
            'data_parallel_size': {
                'type': 'integer',
                'min': 1,
                'default': 1
            },
            'horizontal_parallel_size': {
                'type': 'integer',
                'min': 1,
                'default': 1
            },
            'pipeline_parallel' : {
                'type': 'dict',
                'default_setter': lambda _: {},
                'required': False,
                'schema': {
                    'pipeline_parallel_size': {
                        'type': 'integer',
                        'min': 1,
                        'default': 1
                    },
                    'num_pipeline_micro_batches': {
                        'type': 'integer',
                        'min': 1,
                        'default': 1
                    },
                    'pipeline_cut_info_string': {
                        'type': 'string',
                        'default': ''
                    },
                    'sliced_schema': {
                        'type': 'dict',
                        'default_setter': lambda _: {},
                        'keysrules': {'type': 'string'},
                        'valuesrules': {
                            'type': 'list',
                            'schema': {'type': 'integer'}
                        }
                    },
                    'sliced_axes': {
                        'type': 'dict',
                        'default_setter': lambda _: {},
                        'keysrules': {'type': 'string'},
                        'valuesrules': {'type': 'integer'}
                    },
                    'sliced_tensor_names': {
                        'type': 'list',
                        'schema': {'type': 'string'},
                        'default': []
                    }
                }
            },
            'allreduce_post_accumulation': {
                'type': 'boolean',
                'default': False
            },
            'deepspeed_zero_optimization': {
                'type': 'dict',
                'default_setter': lambda _: {},
                'required': False,
                'schema': {
                    'stage': {
                        'type': 'integer',
                        'min': 0,
                        'max': 1,
                        'default': 0
                    },
                }
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
        'default_setter': lambda _: {},
        'required': False,
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
    'graph_transformer': {
        'type': 'dict',
        'default_setter': lambda _: {},
        'required': False,
        'schema': {
            'attn_dropout_recompute': {
                'type': 'boolean',
                'default': False
            },
            'gelu_recompute': {
                'type': 'boolean',
                'default': False
            },
            'transformer_layer_recompute': {
                'type': 'boolean',
                'default': False
            },
            'number_recompute_layers': {
                'type': 'integer',
                'min': 0,
                'default': 0
            },
            'allow_layer_norm_mod_precision': {
                'type': 'boolean',
                'default': False
            },
            'propagate_cast_ops_config': {
                'type': 'dict',
                'default_setter': lambda _: {},
                'required': False,
                'schema': {
                    'strategy': {
                        'type': 'propagate_cast_ops_strategy',
                        'nullable': True,
                        'min': PropagateCastOpsStrategy.INSERT_AND_REDUCE,
                        'max': PropagateCastOpsStrategy.FLOOD_FILL,
                        'default': PropagateCastOpsStrategy.INSERT_AND_REDUCE
                    },
                    'level': {
                        'type': 'integer',
                        'min': -1,
                        'default': -1
                    },
                    'allow': {
                        'type': 'list',
                        'schema': {'type': 'string'},
                        'default': []
                    }
                }
           }
        }
    },
    'utils': {
        'type': 'dict',
        'default_setter': lambda _: {},
        'required': False,
        'schema': {
            'frozen_weights': {
                'type': 'list',
                'default': []
            },
            'grad_norm_clip': {
                'type': 'boolean',
                'default': True
            },
            'invertible_layer_norm_gradient': {
                'type': 'boolean',
                'default': False
            },
            'run_symbolic_shape_infer': {
                'type': 'boolean',
                'default': False
            }
        }
    },
    'debug': {
        'type': 'dict',
        'default_setter': lambda _: {},
        'required': False,
        'schema': {
            'deterministic_compute': {
                'type': 'boolean',
                'default': False
            },
            'check_model_export': {
                'type': 'boolean',
                'default': False
            },
            'graph_save_paths' : {
                'type' : 'dict',
                'default_setter': lambda _: {},
                'required': False,
                'schema': {
                    'model_after_graph_transforms_path': {
                        'type': 'string',
                        'default': ''
                    },
                    'model_with_gradient_graph_path':{
                        'type': 'string',
                        'default': ''
                    },
                    'model_with_training_graph_path': {
                        'type': 'string',
                        'default': ''
                    },
                    'model_with_training_graph_after_optimization_path': {
                        'type': 'string',
                        'default': ''
                    },
                }
            },
        }
    },
    '_internal_use': {
        'type': 'dict',
        'default_setter': lambda _: {},
        'required': False,
        'schema': {
            'enable_internal_postprocess': {
                'type': 'boolean',
                'default': True
            },
            'extra_postprocess': {
                'check_with': _check_is_callable,
                'nullable': True,
                'default': None
            },
            'onnx_opset_version': {
                'type': 'integer',
                'min': 12,
                'max': 13,
                'default': 12
            },
            'enable_onnx_contrib_ops': {
                'type': 'boolean',
                'default': True
            }
        }
    },
    'provider_options':{
        'type': 'dict',
        'default_setter': lambda _: {},
        'required': False,
        'allow_unknown': True,
        'schema': {}
    },
    'session_options': {
        'type': 'session_options',
        'nullable': True,
        'default': None
    },
}
