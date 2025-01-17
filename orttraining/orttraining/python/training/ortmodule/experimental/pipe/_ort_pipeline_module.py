# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib.metadata
from functools import partial

import torch.nn as nn
from deepspeed.pipe import LayerSpec, PipelineModule, TiedLayerSpec
from deepspeed.runtime import utils as ds_utils
from deepspeed.runtime.activation_checkpointing import checkpointing
from packaging.version import Version

from onnxruntime.training.ortmodule import DebugOptions, ORTModule

# Check if DeepSpeed is installed and meets the minimum version requirement
minimum_version = Version("0.9.0")
installed_version = Version(importlib.metadata.version("deepspeed"))

if installed_version < minimum_version:
    raise ImportError(f"DeepSpeed >= {minimum_version} is required, but {installed_version} is installed.")


class ORTPipelineModule(PipelineModule):
    """ORTPipelineModule pipeline module.

    A customized version of DeepSpeed's PipelineModule that wraps each neural network layer
    with ONNX Runtime's ORTModule. This modification allows leveraging ONNX Runtime optimizations
    for the forward and backward passes, potentially enhancing execution performance and efficiency.

    Please locate the "Using ORTPipelineModule for Deepspeed Pipeline Parallel" section in the "docs/ORTModule_Training_Guidelines.md" file of the ORT repository for more information.

    .. note::
        Pipeline parallelism is not compatible with ZeRO-2 and ZeRO-3.

    Args:
        layers (Iterable): A sequence of layers defining pipeline structure. Can be a ``torch.nn.Sequential`` module.
        num_stages (int, optional): The degree of pipeline parallelism. If not specified, ``topology`` must be provided.
        topology (``deepspeed.runtime.pipe.ProcessTopology``, optional): Defines the axes of parallelism axes for training. Must be provided if ``num_stages`` is ``None``.
        loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
        seed_layers(bool, optional): Use a different seed for each layer. Defaults to False.
        seed_fn(type, optional): The custom seed generating function. Defaults to random seed generator.
        base_seed (int, optional): The starting seed. Defaults to 1234.
        partition_method (str, optional): The method upon which the layers are partitioned. Defaults to 'parameters'.
        activation_checkpoint_interval (int, optional): The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.
        activation_checkpoint_func (callable, optional): The function to use for activation checkpointing. Defaults to ``deepspeed.checkpointing.checkpoint``.
        checkpointable_layers(list, optional): Checkpointable layers may not be checkpointed. Defaults to None which does not additional filtering.
        debug_options(onnxruntime.training.ortmodule.DebugOptions): An instance of onnxruntime.training.ortmodule.DebugOptions or None.
            If provided, it will be used to configure debugging options for ORTModule, This is done so we can add the name of the layer to avoid overwriting the ONNX files.
    """

    def __init__(
        self,
        layers,
        num_stages=None,
        topology=None,
        loss_fn=None,
        seed_layers=False,
        seed_fn=None,
        base_seed=1234,
        partition_method="parameters",
        activation_checkpoint_interval=0,
        activation_checkpoint_func=checkpointing.checkpoint,
        checkpointable_layers=None,
        debug_options=None,
    ):
        """
        Initialize the ORTPipelineModule with the option to include ONNX Runtime debug options.
        """

        self.ort_kwargs = {"debug_options": debug_options} if debug_options is not None else {}

        super().__init__(
            layers,
            num_stages,
            topology,
            loss_fn,
            seed_layers,
            seed_fn,
            base_seed,
            partition_method,
            activation_checkpoint_interval,
            activation_checkpoint_func,
            checkpointable_layers,
        )

    def _build(self):
        """
        This method does the same thing as PipelineModule._build() method, the only difference is that it wraps each layer with ORTModule.
        It also handles saving ONNX models with debug options in case of exporting multiple models.
        """
        specs = self._layer_specs

        for local_idx, layer in enumerate(specs[self._local_start : self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    ds_utils.set_random_seed(self.base_seed + layer_idx)

            # Recursively build PipelineModule objects
            if isinstance(layer, PipelineModule):
                raise NotImplementedError("RECURSIVE BUILD NOT YET IMPLEMENTED")

            # TODO: Support wrapping for LayerSpec and TiedLayerSpec in addition to nn.Module in sequential.
            # Currently, we only support wrapping nn.Module instances.

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, nn.Module):
                name = str(layer_idx)

                if "debug_options" in self.ort_kwargs:
                    new_onnx_prefix = name + "_" + self.ort_kwargs["debug_options"].onnx_prefix
                    parallel_debug_options = DebugOptions(
                        self.ort_kwargs["debug_options"].log_level,
                        self.ort_kwargs["debug_options"].save_onnx,
                        new_onnx_prefix,
                    )
                    wrapped_layer = ORTModule(layer, parallel_debug_options)
                else:
                    wrapped_layer = ORTModule(layer)

                self.forward_funcs.append(wrapped_layer)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, wrapped_layer)

            # TiedLayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, TiedLayerSpec):
                # Build and register the module if we haven't seen it before.
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build()
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr

                if layer.forward_fn is None:
                    # Just use forward()
                    self.forward_funcs.append(self.tied_modules[layer.key])
                else:
                    # User specified fn with args (module, input)
                    self.forward_funcs.append(partial(layer.forward_fn, self.tied_modules[layer.key]))

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)

                if "debug_options" in self.ort_kwargs:
                    new_onnx_prefix = name + "_" + self.ort_kwargs["debug_options"].onnx_prefix
                    parallel_debug_options = DebugOptions(
                        self.ort_kwargs["debug_options"].log_level,
                        self.ort_kwargs["debug_options"].save_onnx,
                        new_onnx_prefix,
                    )
                    module = ORTModule(module, parallel_debug_options)
                else:
                    module = ORTModule(module)

                self.forward_funcs.append(module)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, module)

            # Last option: layer may be a functional (e.g., lambda). We do nothing in
            # that case and just use it in forward()
            else:
                self.forward_funcs.append(layer)

        # All pipeline parameters should be considered as model parallel in the context
        # of our FP16 optimizer
        for p in self.parameters():
            p.ds_pipe_replicated = False
