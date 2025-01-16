# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import copy
import inspect
import io
import logging
import os
from collections import OrderedDict
from functools import partial
from hashlib import md5 as hash_fn
from typing import Mapping, Sequence

import onnx
import torch

from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.training.utils import (
    ORTModelInputOutputSchemaType,
    ORTModelInputOutputType,
    PrimitiveType,
    onnx_dtype_to_pytorch_dtype,
    unflatten_data_using_schema,
)

from . import _io, _utils, export_context
from ._fallback import ORTModuleDeviceException, ORTModuleIOError, ORTModuleONNXModelException, wrap_exception
from ._logger import LogColor, LogLevel, ORTModuleInitPhase, SuppressLogs, TimeTracker, TrackTimeForStaticFunction
from ._onnx_models import _get_onnx_file_name, _save_model
from ._runtime_inspector import FlagAndPrintDensity, RuntimeInspector
from ._utils import check_function_has_param, get_rank
from ._zero_stage3_compatibility import stage3_export_context
from .options import DebugOptions, _RuntimeOptions


class ExportedModelInfo:
    """Encapsulates the information of the exported model.

    After ONNX model export, the model info is collected and encapsulated in this class, including:
    1. The ONNX graph input names.
    2. Graph input requiring gradient information.
    3. The model's forward function signature and args/kwargs schema, used as a cache key to compare with the current
         inputs to see if the model needs to be re-exported.

    This data structure is returned by the GraphTransitionManager._export_model method.

    """

    def __init__(
        self,
        module_forward_args_schema: ORTModelInputOutputSchemaType,
        module_forward_kwargs_schema: ORTModelInputOutputSchemaType,
        onnx_graph_input_names: list[str],
        onnx_graph_input_names_require_grad: list[str],
        onnx_graph_input_names_user_defined: list[str],
        onnx_graph_input_names_require_grad_user_defined: list[str],
        exported_model: onnx.ModelProto,
        module_forward_output_schema: ORTModelInputOutputSchemaType,
    ):
        # Used as a baseline to compare with the current inputs (args/kwargs) to see if the model needs to be re-exported.
        self.module_forward_args_schema: ORTModelInputOutputSchemaType | None = module_forward_args_schema
        self.module_forward_kwargs_schema: ORTModelInputOutputSchemaType | None = module_forward_kwargs_schema

        # Input names parsed and then flatten from the model's forward function signature + buffers + parameters (since we use
        # keep_initializers_as_inputs=True for model export)
        # Be noted: all inputs are used by the model for its compute.
        self.onnx_graph_input_names: list[str] = copy.deepcopy(onnx_graph_input_names)

        # A subset of onnx_graph_input_names.
        # Input names that require gradient parsed and then flatten from the model's forward function signature
        # This should contain both the user-defined input names, the buffer names, and the parameter names (since we use
        # keep_initializers_as_inputs=True for model export)
        # Be noted: all inputs are used by the model for its compute.
        self.onnx_graph_input_names_require_grad: list[str] = copy.deepcopy(onnx_graph_input_names_require_grad)

        # Input names parsed from the model's forward function signature.
        # Be noted: all inputs are used by the model for its compute.
        # The ONNX graph input names exclude the parameters, and buffers.
        self.onnx_graph_input_names_user_defined = copy.deepcopy(onnx_graph_input_names_user_defined)

        # A subset of onnx_graph_input_names_user_defined.
        self.onnx_graph_input_names_require_grad_user_defined = copy.deepcopy(
            onnx_graph_input_names_require_grad_user_defined
        )

        # Exported model proto.
        self.exported_model: onnx.ModelProto | None = exported_model

        # Used for unflattening the outputs from the ORT forward run.
        self.module_forward_output_schema: ORTModelInputOutputSchemaType | None = module_forward_output_schema

    def __str__(self):
        return f"""ExportedModelInfo class:
            \tonnx_graph_input_names: {self.onnx_graph_input_names}
            \tonnx_graph_input_names_require_grad: {self.onnx_graph_input_names_require_grad}
            \tmodule_forward_args_schema: {self.module_forward_args_schema}
            \tmodule_forward_kwargs_schema: {self.module_forward_kwargs_schema}
            \tmodule_forward_output_schema: {self.module_forward_output_schema}
        """

    def __repr__(self):
        return self.__str__()


class PostExportProcessedModelInfo:
    """Encapsulates the information of the post-export processed model.

    After ONNX model post-export processing, the model info is collected and encapsulated in this class, including:
    1. The ONNX graph input names, dynamic axes, and input data accessor functions.
    2. Graph input requiring gradient information.
    3. The interface to construct the inputs for the ORT forward run, from original given inputs running for PyTorch.
    4. The interface to restore the outputs from the ORT forward run, back to the original data structure.

    """

    def __init__(
        self,
        flatten_module: torch.nn.Module,
        onnx_graph_input_names_user_defined: list[str],
        onnx_graph_input_names_require_grad_user_defined: list[str],
        onnx_graph_input_names: list[str],
        onnx_graph_input_names_require_grad: list[str],
        onnx_graph_input_dynamic_axes_map: dict[str, dict[int, str]],
        module_forward_output_schema: ORTModelInputOutputSchemaType,
        post_export_processed_model: onnx.ModelProto,
        onnx_graph_input_data_accessor_user_defined: dict[str, callable],
        onnx_graph_input_const_as_tensor: dict[str, torch.device],
        enable_mem_efficient_grad_management: bool,
    ):
        self._flattened_module = flatten_module

        # Input names parsed from the model's forward function signature.
        # Be noted: all inputs are used by the model for its compute.
        # The ONNX graph input names exclude the parameters, and buffers.
        self.onnx_graph_input_names_user_defined = copy.deepcopy(onnx_graph_input_names_user_defined)

        # A subset of onnx_graph_input_names_user_defined.
        self.onnx_graph_input_names_require_grad_user_defined = copy.deepcopy(
            onnx_graph_input_names_require_grad_user_defined
        )

        # Input names for the pre-gradient-build graph.
        # This may be different with the one in ExportedGraph since we may modify the graph inputs as needed
        # for example when memory efficient gradient management is enabled.
        self.onnx_graph_input_names: list[str] = copy.deepcopy(onnx_graph_input_names)

        # A subset of onnx_graph_input_names.
        # Input names that require gradients for the pre-gradient-build graph.
        self.onnx_graph_input_names_require_grad: list[str] = copy.deepcopy(onnx_graph_input_names_require_grad)

        # Create symbolic names for each dimension of the graph input (e.g. onnx_graph_input_names).
        # The key is the input name, the value is a dict of {dim_index: symbolic_dim_name}
        # e.g. {"input1": {0: "input1_dim0", 1: "input1_dim1"}, "input2": {0: "input2_dim0"}}
        self.onnx_graph_input_dynamic_axes_map: dict[str, dict[int, str]] = onnx_graph_input_dynamic_axes_map

        self._post_export_processed_model: onnx.ModelProto | None = post_export_processed_model

        # A function to access the input data from the args and kwargs.
        # If it is not None, the length is same as onnx_graph_input_names_user_defined.
        # For i-th input name, we can use the i-th function to get the input data from args and kwargs.
        self.onnx_graph_input_data_accessor_user_defined: dict[str, callable] | None = (
            onnx_graph_input_data_accessor_user_defined
        )

        self.onnx_graph_input_const_as_tensor: dict[str, torch.device] | None = onnx_graph_input_const_as_tensor

        self.is_mem_efficient_grad_management_enabled = enable_mem_efficient_grad_management

        # Used for unflattening the outputs from the ORT forward run.
        self.module_forward_output_schema: ORTModelInputOutputSchemaType | None = module_forward_output_schema

        # A buffer to hold the inputs for the ORT forward run. For performance, we reuse the same buffer for each run.
        self._buffer_for_ort_runs: dict[str, torch.Tensor] | None = None

    def __str__(self):
        return f"""PostExportProcessedModelInfo class:
            \tonnx_graph_input_names: {self.onnx_graph_input_names}
            \tonnx_graph_input_names_require_grad: {self.onnx_graph_input_names_require_grad}
            \tonnx_graph_input_dynamic_axes_map: {self.onnx_graph_input_dynamic_axes_map}
            \tonnx_graph_input_names_user_defined: {self.onnx_graph_input_names_user_defined}
            \tonnx_graph_input_names_require_grad_user_defined: {self.onnx_graph_input_names_require_grad_user_defined}
            \tbuffer_for_ort_runs.keys(): {self._buffer_for_ort_runs.keys() if self._buffer_for_ort_runs else None}
        """

    def __repr__(self):
        return self.__str__()

    def construct_inputs(
        self,
        args: Sequence[ORTModelInputOutputType],
        kwargs: Mapping[str, ORTModelInputOutputType],
        constant_as_tensor: bool,
        device: torch.device,
    ):
        """Constructs the inputs for the forward method

        The inputs are constructed in the order they appear in the model's forward function signature
        """
        from ._mem_efficient_grad_mgmt import (
            MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME,
            MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_DTYPE,
            MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_SHAPE,
        )

        # First time construct the buffer for the ORT forward run.
        if self._buffer_for_ort_runs is None:
            self._buffer_for_ort_runs = OrderedDict()

            # Create the buffers for the inputs that are either parameters or buffers in the original module.
            # For user inputs, fill with None for now, and will be filled dynamically during the forward run.

            parameter_names = {k: v for k, v in self._flattened_module.named_parameters()}
            buffer_names = {k: v for k, v in self._flattened_module.named_buffers()}

            for input_name in self.onnx_graph_input_names:
                if input_name in parameter_names:
                    self._buffer_for_ort_runs[input_name] = parameter_names[input_name]
                elif input_name in buffer_names:
                    self._buffer_for_ort_runs[input_name] = buffer_names[input_name]
                else:
                    self._buffer_for_ort_runs[input_name] = (
                        None  # Fill None for user input first, will be overridden later.
                    )

        for name in self.onnx_graph_input_names_user_defined:
            if self.is_mem_efficient_grad_management_enabled and name == MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME:
                self._buffer_for_ort_runs[name] = torch.zeros(
                    MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_SHAPE,
                    dtype=onnx_dtype_to_pytorch_dtype(MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_DTYPE),
                    device=device,
                ).requires_grad_()
                continue

            if name in self.onnx_graph_input_data_accessor_user_defined:
                assert name in self._buffer_for_ort_runs, f"{name} is not in buffer_for_ort_runs"
                data = self.onnx_graph_input_data_accessor_user_defined[name](args, kwargs)
                if name in self.onnx_graph_input_const_as_tensor:
                    data = PrimitiveType.get_tensor(data, device)
                self._buffer_for_ort_runs[name] = data
            else:
                raise wrap_exception(
                    ORTModuleONNXModelException,
                    RuntimeError(f"Input is present in ONNX graph but not provided: {name}."),
                )

        return self._buffer_for_ort_runs

    def restore_outputs(self, ort_flatten_outputs: list[torch.Tensor]):
        """Restores the outputs from the ORT forward run, back to the original data structure"""

        try:
            return unflatten_data_using_schema(ort_flatten_outputs, self.module_forward_output_schema)
        except TypeError as e:
            raise wrap_exception(
                ORTModuleIOError,
                TypeError(f"ORTModule fails to unflatten user output: {e}"),
            ) from None


class GraphTransitionManager:
    """Manage the graph transition from 1). PyTorch to ONNX export and 2). ONNX to ONNX post-export processing."""

    def __init__(
        self,
        flatten_module: torch.nn.Module,
        export_mode: int,
        debug_options: DebugOptions,
        runtime_options: _RuntimeOptions,
        time_tracker: TimeTracker,
        runtime_inspector: RuntimeInspector,
        logger: logging.Logger,
    ):
        self._device = _utils._get_device_from_module(flatten_module)
        self._export_mode = export_mode

        self._debug_options = debug_options
        self._runtime_options = runtime_options

        self._export_extra_kwargs = {}

        self._logger = logger

        # Tracker for ORTModule model export.
        self._time_tracker = time_tracker

        self._runtime_inspector = runtime_inspector

        # A signal to indicate if the original model has changed and need a re-export.
        self._original_model_has_changed = False

        self._flatten_module = flatten_module

        # Forward function input parameters of the original module.
        self._module_forward_func_parameters: list[inspect.Parameter] = list(
            inspect.signature(self._flatten_module._original_module.forward).parameters.values()
        )
        # TODO: remove after PyTorch ONNX exporter supports VAR_KEYWORD parameters.
        for input_parameter in self._module_forward_func_parameters:
            if input_parameter.kind == inspect.Parameter.VAR_KEYWORD:
                logger.info("The model's forward method has **kwargs parameter which has EXPERIMENTAL support!")

        # Model info collected from the original module's forward function signature and args/kwargs, used for ONNX export.
        self._model_info_for_export: _io.ModelInfoForExport | None = None
        self._exported_model_info: ExportedModelInfo | None = None

        # Model info after export and post export processing.
        self._post_export_processed_model_info = None

    def get_post_processed_model(
        self, args: Sequence[ORTModelInputOutputType], kwargs: Mapping[str, ORTModelInputOutputType]
    ) -> tuple[bool, PostExportProcessedModelInfo]:
        """Check if the post-export processed ONNX model can be reused, otherwise, reconstruct the model.

        Return True if the model can be reused, otherwise, return False.
        The model can be reused when the following conditions are met:
            a. The model has been exported before, and the inputs (args/outputs) schemas are the same as the previous ones.
            b. If it is in training mode, the graph inputs requiring gradient are the same as the previous ones.

        """

        if self._device is None:
            device = _utils.get_device_from_module_and_inputs(self._flatten_module._original_module, args, kwargs)
            if not self._device or self._device != device:
                self._device = device
                if not self._device:
                    raise wrap_exception(
                        ORTModuleDeviceException, RuntimeError("A device must be specified in the model or inputs!")
                    )

        # Extract the schema from the args and kwargs, and compare it with the pre-exported one if already exported.
        cur_model_info_for_export = _io.parse_inputs_for_onnx_export(
            self._module_forward_func_parameters,
            args,
            kwargs,
            True,
            self._device,
            self._export_mode,
            self._logger,
            self._export_extra_kwargs,
        )

        need_export_model = GraphTransitionManager._export_check(
            prev_exported_model_info=self._exported_model_info,
            original_model_has_changed=self._original_model_has_changed,
            cur_args_schema=cur_model_info_for_export.onnx_graph_input_arg_schema,
            cur_kwargs_schema=cur_model_info_for_export.onnx_graph_input_kwarg_schema,
            logger=self._logger,
        )

        if need_export_model:
            # Note related to the _io.FlattenedModule export!!!
            #
            # The _io.FlattenedModule serves as a module wrapper designed to support tuple inputs and outputs for
            # PyTorch run during ONNX export. (Remember the PyTorch exporter handles tuple inputs and outputs better.)
            # Internally, it facilitates the acceptance of tuple inputs and the generation of tuple outputs by invoking
            # the original module's forward function. The workflow involves the following steps:

            # 1. Prior to export, both args and kwargs are flattened into a 1-D tensor list, and schemas for the
            #    flattened args and kwargs are generated. This schemas are essential for the subsequent un-flattening
            #    process.

            # 2. The flattened inputs (args + kwargs) are passed to the _io.FlattenedModule's forward run.

            # 3. The args schema and kwargs schema, etc are conveyed to the _io.FlattenedModule by setting the
            #    corresponding attributes.

            # 4. Within the _io.FlattenedModule's forward run, the inputs are un-flattened to the original args and
            #    kwargs using the associated schemas, and then they are passed to the original module's forward function.

            # 5. Upon the completion of the forward function, the outputs from the original module are flattened and
            # returned to the caller.

            # 6. The 1-D flattened output tensors retain the same order as the outputs from the ONNX Runtime (ORT)
            #    forward run. To facilitate un-flattening during subsequent ORT runs, the output schema is saved as
            #    an attribute named `_output_schema` in the _io.FlattenedModule.

            copied_args = copy.copy(args)
            copied_kwargs = copy.copy(kwargs)
            flatten_inputs = []

            # This looks a bit duplicated with `extract_data_and_schema` function, but this might be better to
            # defined as a specialized logic that is the counter-part of `parse_inputs_for_onnx_export`, which handles
            # args and kwargs separately.
            for name, data_accessor in cur_model_info_for_export.onnx_graph_input_data_accessor_user_defined.items():
                d = data_accessor(copied_args, copied_kwargs)
                if name in cur_model_info_for_export.onnx_graph_input_const_as_tensor:
                    flatten_inputs.append(
                        PrimitiveType.get_tensor(
                            d,
                            cur_model_info_for_export.onnx_graph_input_const_as_tensor[name],
                        )
                    )
                else:
                    if isinstance(d, torch.Tensor):
                        flatten_inputs.append(d)

                    # Ignore all other non-tensor inputs.

            self._flatten_module._device = self._device
            self._flatten_module._args_schema = cur_model_info_for_export.onnx_graph_input_arg_schema
            self._flatten_module._kwargs_schema = cur_model_info_for_export.onnx_graph_input_kwarg_schema
            self._flatten_module._num_positionals = cur_model_info_for_export.num_positional_args

            self._logger.info(f"do_export started, model info for export: {cur_model_info_for_export}")

            (
                exported_model,
                module_output_schema,  # Retrieved from _io.FlattenedModule's _output_schema
                onnx_graph_input_names,
                onnx_graph_input_names_require_grad,
            ) = GraphTransitionManager._export_model(
                flattened_module=self._flatten_module,
                model_info_for_export=cur_model_info_for_export,
                flatten_module_inputs=flatten_inputs,
                deepcopy_before_model_export=self._runtime_options.deepcopy_before_model_export,
                device=self._device,
                ortmodule_cache_dir=self._runtime_options.ortmodule_cache_dir,
                enable_custom_autograd_function=self._runtime_options.enable_custom_autograd_function,
                enable_zero_stage3_support=self._runtime_options.enable_zero_stage3_support,
                enable_embedding_sparse_optimizer=self._runtime_options.enable_embedding_sparse_optimizer,
                onnx_opset_version=self._runtime_options.onnx_opset_version,
                stage3_param_handle=self,
                debug_options=self._debug_options,
                time_tracker=self._time_tracker,
                runtime_inspector=self._runtime_inspector,
                logger=self._logger,
            )

            # Get the intersection of all user-defined input names (parsed from forward function signature) and
            # the exported model input names including both user-defined input names and training parameter/buffer names.
            # It is possible some user-defined input names are not in the exported model input names, if it is not used
            # by the model for its compute.
            onnx_graph_input_names_user_defined = [
                input_name
                for input_name in cur_model_info_for_export.onnx_graph_input_names
                if input_name in onnx_graph_input_names
            ]
            onnx_graph_input_names_require_grad_user_defined = [
                input_name
                for input_name in cur_model_info_for_export.onnx_graph_input_names_require_grad
                if input_name in onnx_graph_input_names_require_grad
            ]

            self._exported_model_info = ExportedModelInfo(
                module_forward_args_schema=cur_model_info_for_export.onnx_graph_input_arg_schema,
                module_forward_kwargs_schema=cur_model_info_for_export.onnx_graph_input_kwarg_schema,
                onnx_graph_input_names=onnx_graph_input_names,
                onnx_graph_input_names_require_grad=onnx_graph_input_names_require_grad,
                onnx_graph_input_names_user_defined=onnx_graph_input_names_user_defined,
                onnx_graph_input_names_require_grad_user_defined=onnx_graph_input_names_require_grad_user_defined,
                exported_model=exported_model,
                module_forward_output_schema=module_output_schema,
            )

            self._model_info_for_export = cur_model_info_for_export

            # Reset the signal to indicate the original model has changed.
            self._original_model_has_changed = False

            # Save the exported model
            if self._debug_options.save_onnx_models.save:
                _save_model(
                    self._exported_model_info.exported_model,
                    os.path.join(
                        self._debug_options.save_onnx_models.path,
                        _get_onnx_file_name(
                            self._debug_options.save_onnx_models.name_prefix, "torch_exported", self._export_mode
                        ),
                    ),
                )

            self._logger.info(f"do_export completed, exported graph infos: {self._exported_model_info}")

        need_re_processed = False
        if need_export_model:
            need_re_processed = True
        else:
            need_re_processed, updated_onnx_graph_input_requires_grads = GraphTransitionManager._reprocess_check(
                flatten_module=self._flatten_module,
                exported_model_info=self._exported_model_info,
                export_mode=self._export_mode,
                model_info_for_export=self._model_info_for_export,
                args=args,
                kwargs=kwargs,
            )
            if need_re_processed:
                # Update the onnx_graph_input_names_require_grads to make it a new default baseline to compare
                # using new iteration data.
                self._exported_model_info.onnx_graph_input_names_require_grad = updated_onnx_graph_input_requires_grads

        if need_re_processed:
            # At this point, the exported model is ready, and we can start post-export processing.
            self._post_export_processed_model_info = GraphTransitionManager._post_export_process(
                flatten_module=self._flatten_module,
                export_mode=self._export_mode,
                exported_model_info=self._exported_model_info,
                model_info_for_export=self._model_info_for_export,
                enable_custom_autograd_function=self._runtime_options.enable_custom_autograd_function,
                enable_zero_stage3_support=self._runtime_options.enable_zero_stage3_support,
                run_symbolic_shape_infer=self._runtime_options.run_symbolic_shape_infer,
                stage3_param_handle=self,
                enable_mem_efficient_grad_management=self._export_mode != torch.onnx.TrainingMode.EVAL
                and self._runtime_options.enable_mem_efficient_grad_management,
                logger=self._logger,
            )

            # Save the post_processed model
            if self._debug_options.save_onnx_models.save:
                _save_model(
                    self._post_export_processed_model_info._post_export_processed_model,
                    os.path.join(
                        self._debug_options.save_onnx_models.path,
                        _get_onnx_file_name(
                            self._debug_options.save_onnx_models.name_prefix, "post_processed", self._export_mode
                        ),
                    ),
                )

        return need_re_processed, self._post_export_processed_model_info

    @staticmethod
    def _export_check(
        prev_exported_model_info: ExportedModelInfo | None,
        original_model_has_changed: bool,
        cur_args_schema: ORTModelInputOutputSchemaType,
        cur_kwargs_schema: ORTModelInputOutputSchemaType,
        logger: logging.Logger,
    ):
        """Check if the model needs to be exported, if yes, return True.

        If either of the following conditions is met, return True:
            1. The model has never been exported before.
            2. The original_model_has_changed is True.
            3. The model input schema parsed from args and kwargs has changed.
        """

        need_export_model = prev_exported_model_info is None  # never exported before

        need_export_model = need_export_model or original_model_has_changed

        need_export_model = (
            need_export_model
            or cur_args_schema != prev_exported_model_info.module_forward_args_schema
            or cur_kwargs_schema != prev_exported_model_info.module_forward_kwargs_schema
        )

        logger.info(f"_export_check completed - need_export_model: {need_export_model}")

        return need_export_model

    @staticmethod
    def _reprocess_check(
        flatten_module: _io._FlattenedModule,
        exported_model_info: ExportedModelInfo,
        export_mode: int,
        model_info_for_export: _io.ModelInfoForExport,
        args: Sequence[ORTModelInputOutputType],
        kwargs: Mapping[str, ORTModelInputOutputType],
    ) -> bool:
        """Check if the exported model needs to be re-processed, if yes,
        return True and the updated onnx_graph_input_requires_grads.

        For the following cases, return True:
            1. The export mode is TRAINING and the model's input names (including both user input and module parameters)
               requiring gradient change.
        """
        if export_mode == torch.onnx.TrainingMode.TRAINING:
            # If inputs requiring gradient change from forward to the next, the gradient graph builder
            # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad

            # Reinitialize graph builder if the inputs or initializers requiring gradient have changed.
            # This can happen when the user changes the model parameters after the onnx export.
            # Model may have unused params dropped after export, so we only check those inputs existing in onnx graph.

            onnx_graph_input_requires_grads = []
            parameter_names = {k: v for k, v in flatten_module.named_parameters()}
            for input_name in exported_model_info.onnx_graph_input_names:
                if input_name in exported_model_info.onnx_graph_input_names_user_defined:
                    assert input_name in model_info_for_export.onnx_graph_input_data_accessor_user_defined, (
                        f"{input_name} model_info_for_export.onnx_graph_input_data_accessor_user_defined"
                    )
                    # We assume the data accessor should be the same as the one used for the previous export, because
                    # there is args and kwargs schema check during export check phase.
                    if model_info_for_export.onnx_graph_input_data_accessor_user_defined[input_name](
                        args, kwargs
                    ).requires_grad:
                        onnx_graph_input_requires_grads.append(input_name)
                else:
                    assert input_name in parameter_names, f"{input_name} not exist parameter_names"
                    if parameter_names[input_name].requires_grad:
                        onnx_graph_input_requires_grads.append(input_name)

            if onnx_graph_input_requires_grads == exported_model_info.onnx_graph_input_names_require_grad:
                return False, []
            return True, onnx_graph_input_requires_grads

        return False, []

    @staticmethod
    def _post_export_process(
        flatten_module,
        export_mode,
        exported_model_info: ExportedModelInfo,
        model_info_for_export: _io.ModelInfoForExport,
        enable_custom_autograd_function: bool,
        enable_zero_stage3_support: bool,
        run_symbolic_shape_infer: bool,
        stage3_param_handle: type,
        enable_mem_efficient_grad_management: bool,
        logger: logging.Logger,
    ):
        """Post process the exported model, generate the processed model which will be used for initializing graph builder."""

        # Deepcopy the exported model, in case modification affects the exported model.
        post_processed_model = copy.deepcopy(exported_model_info.exported_model)

        if enable_custom_autograd_function:
            from ._custom_autograd_function_exporter import post_process_enabling_autograd_function

            post_processed_model = post_process_enabling_autograd_function(post_processed_model)

        if run_symbolic_shape_infer:
            # MUST call symbolic shape inference after custom autograd function post-processing is done,
            # Otherwise, there is no ctx output for PythonOp.
            post_processed_model = GraphTransitionManager._infer_shapes(post_processed_model)

        if export_mode == torch.onnx.TrainingMode.TRAINING:
            if enable_zero_stage3_support:
                from ._zero_stage3_compatibility import post_processing_enable_zero_stage3_compat

                post_processed_model = post_processing_enable_zero_stage3_compat(
                    post_processed_model,
                    stage3_param_handle._zero_stage3_param_map,
                    [name for name, _ in flatten_module.named_parameters()],
                )

        onnx_graph_input_names_user_defined = copy.deepcopy(exported_model_info.onnx_graph_input_names_user_defined)
        onnx_graph_input_names_require_grad_user_defined = copy.deepcopy(
            exported_model_info.onnx_graph_input_names_require_grad_user_defined
        )
        onnx_graph_input_names = copy.deepcopy(exported_model_info.onnx_graph_input_names)
        onnx_graph_input_names_require_grad = copy.deepcopy(exported_model_info.onnx_graph_input_names_require_grad)

        if enable_mem_efficient_grad_management:
            # Remove those trainable parameters from graph input, as they will be retrieved from weight pull node.
            from ._mem_efficient_grad_mgmt import get_params_connected_to_pull_param_trigger

            # MUST call this before post_processing_enable_mem_efficient_training, otherwise, the onnx graph input
            # will be modified.
            parameter_not_as_graph_input_names = get_params_connected_to_pull_param_trigger(
                flatten_module.named_parameters(), post_processed_model
            )

            if len(parameter_not_as_graph_input_names) > 0:
                for k in parameter_not_as_graph_input_names:
                    if k in onnx_graph_input_names:
                        onnx_graph_input_names.remove(k)

                    if k in onnx_graph_input_names_require_grad:
                        onnx_graph_input_names_require_grad.remove(k)

                from ._mem_efficient_grad_mgmt import MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME

                # Add mem efficient grad trigger name to require_grad_names, so that it will be included in the gradient graph.
                onnx_graph_input_names_user_defined.append(MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME)
                onnx_graph_input_names_require_grad_user_defined.append(MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME)
                onnx_graph_input_names.append(MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME)
                onnx_graph_input_names_require_grad.append(MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME)

                from ._mem_efficient_grad_mgmt import post_processing_enable_mem_efficient_training

                # Override the options if model is not modified.

                (
                    enable_mem_efficient_grad_management,  # Update the flag to indicate the mem efficient grad management is enabled.
                    post_processed_model,
                    stage3_param_handle._param_trigger_grad,
                ) = post_processing_enable_mem_efficient_training(
                    post_processed_model, flatten_module.named_parameters(), parameter_not_as_graph_input_names
                )

                if run_symbolic_shape_infer:
                    post_processed_model = SymbolicShapeInference.infer_shapes(
                        post_processed_model, auto_merge=True, guess_output_rank=True
                    )

        post_export_processed_model_info = PostExportProcessedModelInfo(
            flatten_module,
            onnx_graph_input_names_user_defined,
            onnx_graph_input_names_require_grad_user_defined,
            onnx_graph_input_names,
            onnx_graph_input_names_require_grad,
            model_info_for_export.onnx_graph_input_dynamic_axes_map,
            exported_model_info.module_forward_output_schema,
            post_processed_model,
            model_info_for_export.onnx_graph_input_data_accessor_user_defined,
            model_info_for_export.onnx_graph_input_const_as_tensor,
            enable_mem_efficient_grad_management,
        )

        logger.info(
            f"_post_export_process completed, post-export processed graph infos: {post_export_processed_model_info}"
        )

        return post_export_processed_model_info

    @staticmethod
    def _infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
        """Infer shapes for the exported model."""

        model = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=True)
        return model

    @staticmethod
    @TrackTimeForStaticFunction(ORTModuleInitPhase.EXPORT)
    @SuppressLogs(ORTModuleInitPhase.EXPORT, is_ort_filter=False)
    def _export_model(
        *,
        flattened_module: torch.nn.Module,
        model_info_for_export: _io.ModelInfoForExport,
        flatten_module_inputs: Sequence[ORTModelInputOutputType],
        deepcopy_before_model_export: bool,
        device: torch.device,
        ortmodule_cache_dir: str,
        enable_custom_autograd_function: bool,
        enable_zero_stage3_support: bool,
        enable_embedding_sparse_optimizer: bool,
        onnx_opset_version: int,
        stage3_param_handle: type,
        debug_options: DebugOptions,
        time_tracker: TimeTracker,  # time_tracker MUST be provided here to support TrackTimeForStaticFunction
        runtime_inspector: RuntimeInspector,
        logger: logging.Logger,
    ) -> tuple[onnx.ModelProto, ORTModelInputOutputSchemaType, list[str], list[str]]:
        # Add hooks to check the sparsity of the embedding and label inputs during the export.
        embedding_hook_handles = GraphTransitionManager._add_check_embedding_sparsity_hook(
            enable_embedding_sparse_optimizer, device, logger, runtime_inspector, flattened_module
        )
        label_hook_handles = GraphTransitionManager._add_check_label_sparsity_hook(
            enable_embedding_sparse_optimizer, logger, runtime_inspector, flattened_module
        )

        # Record random states here and restore later in case any of them gets changed during the export,
        # e.g., some sympy functions in symbolic_shape_infer will change Python's random state.
        random_states = _utils.get_random_states()

        torch_exporter_verbose_log = debug_options.log_level < LogLevel.WARNING
        from onnxruntime.training.utils.hooks._subscriber_manager import no_increase_global_step

        with export_context(), no_increase_global_step():
            exported_model, module_output_schema = GraphTransitionManager._get_exported_model(
                flattened_module=flattened_module,
                model_info_for_export=model_info_for_export,
                flatten_module_inputs=flatten_module_inputs,
                deepcopy_before_model_export=deepcopy_before_model_export,
                device=device,
                ortmodule_cache_dir=ortmodule_cache_dir,
                enable_custom_autograd_function=enable_custom_autograd_function,
                enable_zero_stage3_support=enable_zero_stage3_support,
                onnx_opset_version=onnx_opset_version,
                torch_exporter_verbose_log=torch_exporter_verbose_log,
                stage3_param_handle=stage3_param_handle,
                logger=logger,
            )

        onnx_graph_input_names = [input.name for input in exported_model.graph.input]
        parameter_names = [name for name, _ in flattened_module.named_parameters()]
        onnx_graph_input_names_require_grad = [
            input.name
            for input in exported_model.graph.input
            if input.name in parameter_names or input.name in model_info_for_export.onnx_graph_input_names_require_grad
        ]

        # Restore the recorded random states
        _utils.set_random_states(random_states)

        # Clean up all hooks.
        for hook in embedding_hook_handles:
            hook.remove()

        for hook in label_hook_handles:
            hook.remove()

        return exported_model, module_output_schema, onnx_graph_input_names, onnx_graph_input_names_require_grad

    @staticmethod
    def _get_exported_model(
        flattened_module: torch.nn.Module,
        model_info_for_export: _io.ModelInfoForExport,
        flatten_module_inputs: Sequence[ORTModelInputOutputType],
        deepcopy_before_model_export: bool,
        device: torch.device,
        ortmodule_cache_dir: str,
        enable_custom_autograd_function: bool,
        enable_zero_stage3_support: bool,
        onnx_opset_version: int,
        torch_exporter_verbose_log: bool,
        stage3_param_handle: type,
        logger: logging.Logger,
    ) -> tuple[onnx.ModelProto, ORTModelInputOutputSchemaType]:
        """Exports PyTorch `flattened_module` to ONNX for inferencing or training."""

        need_deep_copy = deepcopy_before_model_export and _io.can_module_be_deep_cloned(flattened_module, device)
        if not need_deep_copy:
            if deepcopy_before_model_export:
                logger.warning(
                    "Since the user requested not to deep copy this model, "
                    "the initial weights may not be preserved and could change slightly during the forward run. "
                    "This could cause a minor difference between the ORTModule and the PyTorch run for the "
                    "first iteration. The computation will proceed as normal, but this should be noted."
                )
            else:
                logger.warning(
                    "Due to the limited GPU memory execution manager does not create a deep copy of this model. "
                    "Therefore, the initial weights might be slightly altered during the forward run. "
                    "This could result in a minor discrepancy between the ORTModule and the PyTorch run for the "
                    "first iteration. The computation will continue as usual, but this should be noted."
                )
        (
            output_names,
            output_dynamic_axes,
            module_output_schema,
        ) = _io.parse_outputs_for_onnx_export_and_extract_schema(
            flattened_module, flatten_module_inputs, logger, need_deep_copy
        )

        # Combine the dynamic axes from inputs and outputs
        dynamic_axes = copy.deepcopy(model_info_for_export.onnx_graph_input_dynamic_axes_map)

        dynamic_axes.update(output_dynamic_axes)

        logger.info("Exporting the PyTorch model to ONNX...")

        # Leverage cached model if available
        cache_dir = ortmodule_cache_dir
        if cache_dir:
            filename = os.path.join(
                cache_dir, f"{hash_fn(str(flattened_module).encode()).hexdigest()}_{get_rank()}.onnx"
            )
            if os.path.exists(cache_dir) and os.path.isfile(filename):
                logger.warning(
                    f"Cached model detected! Cached model will be used to save export and initialization time."
                    f"If you want the model to be re-exported then DELETE {filename}."
                )
                exported_model = onnx.load(filename)
                return exported_model, module_output_schema

        # Export torch.nn.Module to ONNX
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for export.
        kwargs = {}
        sample_inputs_copy, sample_kwargs_copy = _io.deepcopy_model_input(*flatten_module_inputs, **kwargs)
        assert len(sample_kwargs_copy) == 0, "Currently, kwargs are not supported for ONNX export."
        sample_inputs_as_tuple = sample_inputs_copy

        # Ops behaving differently under train/eval mode need to be exported with the
        # correct training flag to reflect the expected behavior.
        # For example, the Dropout node in a model is dropped under eval mode.
        assert model_info_for_export.export_mode is not None, "Please use a concrete instance of ExecutionManager"

        try:
            with (
                torch.no_grad(),
                stage3_export_context(enable_zero_stage3_support, stage3_param_handle, flattened_module),
            ):
                required_export_kwargs = {
                    "input_names": model_info_for_export.onnx_graph_input_names,  # did not contains parameters as its input yet
                    "output_names": output_names,
                    "opset_version": onnx_opset_version,
                    "do_constant_folding": False,
                    "training": model_info_for_export.export_mode,
                    "dynamic_axes": dynamic_axes,
                    "verbose": torch_exporter_verbose_log,
                    "export_params": False,
                    "keep_initializers_as_inputs": True,
                }

                if check_function_has_param(torch.onnx.export, "autograd_inlining"):
                    # From some PyTorch version, autograd_inlining is a valid argument.
                    # We allow it to be True if custom autograd function is disabled (where autograd.Function
                    # anyway is not supported in ONNX until it can be inlined).
                    required_export_kwargs["autograd_inlining"] = not enable_custom_autograd_function

                invalid_args = model_info_for_export.export_extra_kwargs.keys() & required_export_kwargs.keys()

                if len(invalid_args) != 0:
                    error_msg = f"The following PyTorch exporter arguments cannot be specified: '{invalid_args}'."
                    raise RuntimeError(error_msg)

                torch.onnx.export(
                    flattened_module,
                    sample_inputs_as_tuple,
                    f,
                    **required_export_kwargs,
                    **model_info_for_export.export_extra_kwargs,
                )
        except Exception as e:
            message = _utils.get_exception_as_string(e)

            # Special handling when Huggingface transformers gradient checkpoint usage pattern found.
            # For new versions of PyTorch 2, tracing torch.utils.checkpoint.checkpoint will be failed like this:
            #   File "microsoft/phi-2/b10c3eba545ad279e7208ee3a5d644566f001670/modeling_phi.py", line 919, in forward
            #     layer_outputs = self._gradient_checkpointing_func(
            #   File "/site-packages/torch/_compile.py", line 24, in inner
            #     return torch._dynamo.disable(fn, recursive)(*args, **kwargs)
            #   File "/site-packages/torch/_dynamo/eval_frame.py", line 470, in _fn
            #     raise RuntimeError(
            #   RuntimeError: Detected that you are using FX to torch.jit.trace a dynamo-optimized function. This is not supported at the moment.
            if (
                "_gradient_checkpointing_func" in message
                and "Detected that you are using FX to torch.jit.trace a dynamo-optimized function" in message
            ):
                is_ckpt_activation_allowed = int(os.getenv("ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT", "0")) == 1
                notes = (
                    " Your model is running with gradient checkpointing, yet the PyTorch exporter\n"
                    " failed during tracing the graph. Try to enable ORTModule's\n"
                    " gradient checkpointing (a.k.a. Transformer layerwise subgraph recompute)\n"
                    " using `export ORTMODULE_MEMORY_OPT_LEVEL=1` for similar or even better memory efficiency.\n"
                )
                if is_ckpt_activation_allowed:
                    # If the user allows the gradient checkpointing export, we should inform the user to disable it,
                    # to make layerwise recompute work.
                    notes += (
                        " We also notice your setting `export ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=1`,\n"
                        " which enables gradient checkpointing torch.autograd.Functions(s) to export.\n"
                        " To enable ORTModule's layerwise recompute, it needs to be turned OFF by\n"
                        " `export ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=0`.\n"
                    )

                logger.error(
                    f"{LogColor.RED}\n"
                    "******************************** IMPORTANT NOTE *******************************\n"
                    f"{notes}"
                    "*******************************************************************************\n"
                    f"{LogColor.ENDC}\n"
                )

            raise wrap_exception(  # noqa: B904
                ORTModuleONNXModelException,
                RuntimeError(
                    f"There was an error while exporting the PyTorch model to ONNX: "
                    f"\n\n{_utils.get_exception_as_string(e)}"
                ),
            )
        exported_model = onnx.load_model_from_string(f.getvalue())

        # Cache model for future runs
        if cache_dir:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            filename = os.path.join(
                cache_dir, f"{hash_fn(str(flattened_module).encode()).hexdigest()}_{get_rank()}.onnx"
            )
            logger.info(f"Caching model for future runs to {filename}.")
            onnx.save(exported_model, filename)

        return exported_model, module_output_schema

    def signal_model_changed(self):
        """Signals the execution manager to re-export the model on the next forward call"""
        self._original_model_has_changed = True

    @staticmethod
    def _add_check_embedding_sparsity_hook(
        enable_embedding_sparse_optimizer: bool,
        device: torch.device,
        logger: logging.Logger,
        runtime_inspector: RuntimeInspector,
        flattened_module: torch.nn.Module,
    ) -> list:
        """
        Add hook to check embedding sparsity and enable padding elimination if applicable.
        1. Iterate through all modules to find Embedding modules with padding_idx >= 0.
        2. Register forward pre hook to the Embedding module and the hook will check sparsity of the embedding input.
        3. If the sparsity is below a threshold, enable padding elimination by adding FlagAndPrintDensity after the
           output. GraphTransformer of PaddingElimination will check the FlagAndPrintDensity and do the actual
           padding elimination graph modification.
        4. Return the hook handles for later removal.

        """
        if not enable_embedding_sparse_optimizer or device.type != "cuda":
            return []

        def _embedding_hook(name, module, args):
            ebd_input = args[0]
            if ebd_input is None or not isinstance(ebd_input, torch.Tensor):
                logger.warning("Embedding input is not a tensor.")
                return None

            valid_token = torch.count_nonzero(ebd_input - module.padding_idx)
            total_token = ebd_input.numel()
            embed_density = float(valid_token) / float(total_token) * 100

            if embed_density < 90:
                logger.info("Embedding sparsity-based optimization is ON for density: %.0f%%", embed_density)
                runtime_inspector._embedding_module_to_padding_density_map[name] = embed_density
                return FlagAndPrintDensity.apply(args[0], module.padding_idx, "embedding")
            else:
                logger.info("Embedding sparsity-based optimization is OFF for density: %.0f%%", embed_density)
                return None

        embedding_hook_handles = []
        for name, sub_module in flattened_module.named_modules():
            if isinstance(sub_module, torch.nn.modules.sparse.Embedding):
                if sub_module.padding_idx is not None and sub_module.padding_idx >= 0:
                    embedding_hook_handles.append(sub_module.register_forward_pre_hook(partial(_embedding_hook, name)))

        return embedding_hook_handles

    @staticmethod
    def _add_check_label_sparsity_hook(
        enable_label_sparse_optimizer: bool,
        logger: logging.Logger,
        runtime_inspector: RuntimeInspector,
        flattened_module: torch.nn.Module,
    ) -> list:
        """
        Add hook to check label sparsity and enable sceloss compute optimization if applicable.
        1. Register forward pre hook to the sceloss module in the model and the hook will check sparsity of the label input.
        2. If the sparsity is below a threshold, enable sceloss compute optimization by adding FlagAndPrintDensity after the
           output. GraphTransformer of InsertGatherBeforeSceLoss will check the FlagAndPrintDensity and do the actual
           sceloss compute optimization graph modification.

        """
        if not enable_label_sparse_optimizer:
            return None

        def _label_hook(name, module, args):
            label_input = args[1]
            if label_input is None or not isinstance(label_input, torch.Tensor):
                logger.warning("Label input is not a tensor.")
                return None

            valid_token = torch.count_nonzero(label_input - module.ignore_index)
            total_token = label_input.numel()
            label_density = float(valid_token) / float(total_token) * 100

            if label_density < 90:
                logger.info("Label sparsity-based optimization is ON for density: %.0f%%", label_density)
                runtime_inspector._sceloss_module_to_ignore_density_map[name] = label_density
                return (args[0], FlagAndPrintDensity.apply(args[1], module.ignore_index, "label"))
            else:
                logger.info("Label sparsity-based optimization is OFF for density: %.0f%%", label_density)
                return None

        label_check_hook_handles = []
        for name, sub_module in flattened_module.named_modules():
            if isinstance(sub_module, torch.nn.modules.loss.CrossEntropyLoss):
                label_check_hook_handles.append(sub_module.register_forward_pre_hook(partial(_label_hook, name)))

        return label_check_hook_handles
