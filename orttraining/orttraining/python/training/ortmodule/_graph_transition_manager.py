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
from hashlib import md5 as hash_fn
from typing import Mapping, Sequence

import onnx
import torch

from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.training.utils import (
    ORTModelInputOutputSchemaType,
    ORTModelInputOutputType,
    PrimitiveType,
    unflatten_data_using_schema,
)
from onnxruntime.training.utils.torch_io_helper import _TensorStub

from . import _io, _utils
from ._fallback import ORTModuleDeviceException, ORTModuleIOError, ORTModuleONNXModelException, wrap_exception
from ._onnx_models import _get_onnx_file_name, _save_model
from ._utils import check_function_has_param, get_rank
from ._zero_stage3_compatibility import stage3_export_context


class ExportedModelInfo:
    """Encapsulates the information of the exported model."""

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
        # Input names parsed and then flatten from the model's forward function signature + buffers + parameters (since we use
        # keep_initializers_as_inputs=True for model export)
        # Be noted: all inputs are used by the model for its compute.
        self.onnx_graph_input_names: list[str] = onnx_graph_input_names

        # A subset of onnx_graph_input_names.
        # Input names that require gradient parsed and then flatten from the model's forward function signature
        # This should contains both the user input names, the buffer names, and the parameter names (since we use
        # keep_initializers_as_inputs=True for model export)
        # Be noted: all inputs are used by the model for its compute.
        self.onnx_graph_input_names_require_grad: list[str] = onnx_graph_input_names_require_grad

        self.onnx_graph_input_names_user_defined = (
            onnx_graph_input_names_user_defined  # The ONNX graph input names excluding the parameters, buffers.
        )

        # The ONNX graph input names excluding the parameters, buffers.
        self.onnx_graph_input_names_require_grad_user_defined = onnx_graph_input_names_require_grad_user_defined

        # Exported model proto.
        self.exported_model: onnx.ModelProto | None = exported_model

        # Used as a baseline to compare with the current inputs (args/kwargs) to see if the model needs to be re-exported.
        self.module_forward_args_schema: ORTModelInputOutputSchemaType | None = module_forward_args_schema
        self.module_forward_kwargs_schema: ORTModelInputOutputSchemaType | None = module_forward_kwargs_schema

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

    def __repro__(self):
        return self.__str__()


class PostExportProcessedModelInfo:
    """Encapsulates the information of the post-export processed model."""

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
        data_accessor: list[callable],
    ):
        self._flattened_module = flatten_module

        # Input names for the pre-gradient-build graph.
        # This may be different with the one in ExportedGraph since we may modify the graph inputs as needed
        # for example when memory efficient gradient management is enabled.
        self.onnx_graph_input_names: list[str] = onnx_graph_input_names

        # A subset of onnx_graph_input_names.
        # Input names that require gradients for the pre-gradient-build graph.
        self.onnx_graph_input_names_require_grad: list[str] = onnx_graph_input_names_require_grad

        # Create symbolic names for each dimension of the graph input (e.g. onnx_graph_input_names).
        # The key is the input name, the value is a dict of {dim_index: symbolic_dim_name}
        # e.g. {"input1": {0: "input1_dim0", 1: "input1_dim1"}, "input2": {0: "input2_dim0"}}
        self.onnx_graph_input_dynamic_axes_map: dict[str, dict[int, str]] = onnx_graph_input_dynamic_axes_map

        self.buffer_for_ort_runs: dict[str, torch.Tensor] = OrderedDict()
        self.onnx_graph_input_names_user_defined = (
            onnx_graph_input_names_user_defined  # The ONNX graph input names excluding the parameters, buffers.
        )

        # The ONNX graph input names excluding the parameters, buffers.
        self.onnx_graph_input_names_require_grad_user_defined = onnx_graph_input_names_require_grad_user_defined

        self._post_export_processed_model: onnx.ModelProto | None = post_export_processed_model

        # A function to access the input data from the args and kwargs.
        # If it is not None, the length is same as onnx_graph_input_names.
        # For i-th input name, we can use the i-th function to get the input data from args and kwargs.
        self.data_accessor: list[callable] | None = data_accessor

        # Used for unflattening the outputs from the ORT forward run.
        self.module_forward_output_schema: ORTModelInputOutputSchemaType | None = module_forward_output_schema

        # Create the buffers for the inputs that are either parameters or buffers in the original module.
        # For user inputs, fill with None for now, and will be filled dynamically during the forward run.
        parameter_names = {k: v for k, v in self._flattened_module.named_parameters()}
        buffer_names = {k: v for k, v in self._flattened_module.named_buffers()}
        for input_name in self.onnx_graph_input_names:
            if input_name in parameter_names:
                self.buffer_for_ort_runs[input_name] = parameter_names[input_name]
            elif input_name in buffer_names:
                self.buffer_for_ort_runs[input_name] = buffer_names[input_name]
            else:
                self.buffer_for_ort_runs[input_name] = None

    def __str__(self):
        return f"""PostExportProcessedModelInfo class:
            \tonnx_graph_input_names: {self.onnx_graph_input_names}
            \tonnx_graph_input_names_require_grad: {self.onnx_graph_input_names_require_grad}
            \tonnx_graph_input_dynamic_axes_map: {self.onnx_graph_input_dynamic_axes_map}
            \tonnx_graph_input_names_user_defined: {self.onnx_graph_input_names_user_defined}
            \tonnx_graph_input_names_require_grad_user_defined: {self.onnx_graph_input_names_require_grad_user_defined}
            \tbuffer_for_ort_runs.keys(): {self.buffer_for_ort_runs.keys()}
        """

    def __repro__(self):
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

        for name in self.onnx_graph_input_names_user_defined:
            if name in self.data_accessor:
                assert name in self.buffer_for_ort_runs, f"{name} is not in buffer_for_ort_runs"
                data = self.data_accessor[name](args, kwargs)
                if PrimitiveType.is_primitive_type(data) and constant_as_tensor:
                    data = PrimitiveType.get_tensor(data, device)
                self.buffer_for_ort_runs[name] = data
            else:
                raise wrap_exception(
                    ORTModuleONNXModelException,
                    RuntimeError(f"Input is present in ONNX graph but not provided: {name}."),
                )

        return self.buffer_for_ort_runs

    def restore_outputs(self, ort_flatten_outputs: list[torch.Tensor]):
        """Restores the outputs from the ORT forward run, back to the original data structure"""

        try:
            # Need to distinguish between a single output and a tuple (having a single tensor)
            if len(ort_flatten_outputs) == 1 and self.module_forward_output_schema is _TensorStub:
                return ort_flatten_outputs[0]
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
        save: bool,
        save_path: str,
        save_name_prefix: str,
        deepcopy_before_model_export: bool,
        torch_exporter_verbose_log: bool,
        enable_zero_stage3_support: bool,  # TODO(): implement as a plugin
        onnx_opset_version: int,
        enable_custom_autograd_function: bool,  # TODO(): implement as a plugin
        enable_symbolic_shape_infer: bool,
        exported_model_cache_dir: str,
        logger: logging.Logger,
    ):
        self._device = _utils._get_device_from_module(flatten_module)
        self._export_mode = export_mode

        # Debug options
        self._save = save
        self._save_path = save_path
        self._save_name_prefix = save_name_prefix

        # Runtime options
        self._deepcopy_before_model_export = deepcopy_before_model_export
        self._torch_exporter_verbose_log = torch_exporter_verbose_log
        self._enable_zero_stage3_support = enable_zero_stage3_support
        self._onnx_opset_version = onnx_opset_version
        self._enable_custom_autograd_function = enable_custom_autograd_function
        self._run_symbolic_shape_infer = enable_symbolic_shape_infer
        self._ortmodule_cache_dir = exported_model_cache_dir

        self._export_extra_kwargs = {}

        self._logger = logger

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

    def use_cache_or_reconstruct_post_processed_model(
        self, args: Sequence[ORTModelInputOutputType], kwargs: Mapping[str, ORTModelInputOutputType]
    ) -> tuple[bool, PostExportProcessedModelInfo]:
        """Check if the post-export processed ONNX model can be reused, otherwise, reconstruct the model.

        Return True if the model can be reused, otherwise, return False.
        The model can be reused when the following conditions are met:
            a. The model has been exported before, and the inputs (args/outputs) schemas are the same as the previous ones.
            b. In training mode, the graph inputs requiring gradient are the same as the previous ones.

        """

        if self._device is None:
            device = _utils.get_device_from_module_and_inputs(self._flatten_module._original_module, args, kwargs)
            if not self._device or self._device != device:
                self._device = device
                if not self._device:
                    raise wrap_exception(
                        ORTModuleDeviceException, RuntimeError("A device must be specified in the model or inputs!")
                    )

        # Extract the schema from the args and kwargs, and compare with the cached one.
        # This check ideally is not needed as we already have the above check, but it is added as a safeguard.
        flatten_args, cur_args_schema = _io._extract_schema(copy.copy(args), self._device)
        flatten_kwargs, cur_kwargs_schema = _io._extract_schema(copy.copy(kwargs), self._device)

        need_export_model = GraphTransitionManager._export_check(
            prev_model_info_for_export=self._exported_model_info,
            original_model_has_changed=self._original_model_has_changed,
            cur_args_schema=cur_args_schema,
            cur_kwargs_schema=cur_kwargs_schema,
            logger=self._logger,
        )

        if need_export_model:
            # Set the information used to unflatten the inputs during the flatten module forward run.
            # Must be set before calling exporting the model.
            self._flatten_module._device = self._device
            self._flatten_module._args_schema = cur_args_schema
            self._flatten_module._kwargs_schema = cur_kwargs_schema
            self._flatten_module._num_positionals = len(flatten_args)

            flatten_inputs = flatten_args + flatten_kwargs

            # Check graph inputs parsed from the model's forward function signature and current inputs,
            # if they are different, we need to re-export the model.
            cur_model_info_for_export = _io.parse_inputs_for_onnx_export(
                self._module_forward_func_parameters,
                args,
                kwargs,
                True,
                self._device,
                self._export_mode,
                self._export_extra_kwargs,
            )

            # Start exporting the model by passing the 1-D flatten tensor list containing all args plus kwargs.
            (
                exported_model,
                module_output_schema,
                onnx_graph_input_names,
                onnx_graph_input_names_require_grad,
            ) = GraphTransitionManager._export_model(
                flattened_module=self._flatten_module,
                model_info_for_export=cur_model_info_for_export,
                flatten_module_inputs=flatten_inputs,
                run_symbolic_shape_infer=self._run_symbolic_shape_infer,
                deepcopy_before_model_export=self._deepcopy_before_model_export,
                device=self._device,
                ortmodule_cache_dir=self._ortmodule_cache_dir,
                enable_custom_autograd_function=self._enable_custom_autograd_function,
                enable_zero_stage3_support=self._enable_zero_stage3_support,
                onnx_opset_version=self._onnx_opset_version,
                torch_exporter_verbose_log=self._torch_exporter_verbose_log,
                stage3_param_handle=self,
                logger=self._logger,
            )

            # Get the intersection of all user defined input names (parsed from forward function signature) and
            # the exported model input names including both user defined names and parameter/buffer names.
            # It is possible some user defined input names are not in the exported model input names, if it is not used
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
                module_forward_args_schema=cur_args_schema,
                module_forward_kwargs_schema=cur_kwargs_schema,
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
            if self._save:
                _save_model(
                    self._exported_model_info.exported_model,
                    os.path.join(
                        self._save_path,
                        _get_onnx_file_name(self._save_name_prefix, "torch_exported", self._export_mode),
                    ),
                )

            self._logger.info(f"do_export completed, exported graph infos: {self._exported_model_info}")

        need_re_processed = False
        if need_export_model:
            need_re_processed = True
        else:
            need_re_processed, updated_onnx_graph_input_requires_grads = GraphTransitionManager._reprocess_check(
                flatten_module=self._flatten_module,
                prev_exported_model_info=self._exported_model_info,
                export_mode=self._export_mode,
                cur_model_info_for_export=self._model_info_for_export,
            )
            if need_re_processed:
                # Update the onnx_graph_input_names_require_grads to make it a new default baseline to compare
                # using new iteration data.
                self._exported_model_info.onnx_graph_input_names_require_grad = updated_onnx_graph_input_requires_grads

        if need_re_processed:
            # At this point, the exported model is ready, and we can start post-export processing.
            self._post_export_processed_model_info = GraphTransitionManager._post_export_process(
                flatten_module=self._flatten_module,
                device=self._device,
                export_mode=self._export_mode,
                exported_model_info=self._exported_model_info,
                model_info_for_export=self._model_info_for_export,
                enable_custom_autograd_function=self._enable_custom_autograd_function,
                enable_zero_stage3_support=self._enable_zero_stage3_support,
                stage3_param_handle=self,
                logger=self._logger,
            )

            # Save the post_processed model
            if self._save:
                _save_model(
                    self._post_export_processed_model_info._post_export_processed_model,
                    os.path.join(
                        self._save_path,
                        _get_onnx_file_name(self._save_name_prefix, "post_processed", self._export_mode),
                    ),
                )

        return need_re_processed, self._post_export_processed_model_info

    @staticmethod
    def _export_check(
        prev_exported_model_info: ExportedModelInfo,
        original_model_has_changed: bool,
        cur_args_schema: ORTModelInputOutputSchemaType,
        cur_kwargs_schema: ORTModelInputOutputSchemaType,
        logger: logging.Logger,
    ):
        """Check if the model needs to be exported, if yes, return True.

        For the following cases, return True:
            1. The model has never been exported before.
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
        flatten_module, prev_exported_model_info, export_mode, cur_model_info_for_export: _io.ModelInfoForExport
    ) -> bool:
        """Check if the exported model needs to be re-processed, if yes,
        return True and the updated onnx_graph_input_requires_grads.

        For the following cases, return True:
            1. The export mode is TRAINING and the model's input names (including both user input and module parameters)
               requiring gradient change.
        """
        if export_mode == torch.onnx.TrainingMode.TRAINING:
            # If inputs requiring gradient change from forward to the next, the module_gradient_graph_builder
            # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad

            # Reinitialize graph builder if the inputs or initializers requiring gradient have changed.
            # This can happen when the user changes the model parameters after the onnx export.
            # Model may have unused params dropped after export, so we only check those inputs existing in onnx graph.

            onnx_graph_input_requires_grads = []
            parameter_names = {k: v for k, v in flatten_module.named_parameters()}
            for input_name in prev_exported_model_info.onnx_graph_input_names:
                if input_name in parameter_names and parameter_names[input_name].requires_grad:
                    onnx_graph_input_requires_grads.append(input_name)
                else:
                    # If not in parameter list, then it would come from user defined inputs.
                    if input_name in cur_model_info_for_export.onnx_graph_input_names_require_grad:
                        onnx_graph_input_requires_grads.append(input_name)

            if onnx_graph_input_requires_grads == prev_exported_model_info.onnx_graph_input_names_require_grad:
                return False, []
            return True, onnx_graph_input_requires_grads

        return False, []

    @staticmethod
    def _post_export_process(
        flatten_module,
        device,
        export_mode,
        exported_model_info: ExportedModelInfo,
        model_info_for_export: _io.ModelInfoForExport,
        enable_custom_autograd_function: bool,
        enable_zero_stage3_support: bool,
        stage3_param_handle: type,
        logger: logging.Logger,
    ):
        """Post process the exported model, generate the processed model which will be used for initializing graph builder."""

        # Deepcopy the exported model, in case modification affects the exported model.

        # TODO(): Do pre-grad graph modification as needed, for memory efficient gradient management, etc.
        post_processed_model = copy.deepcopy(exported_model_info.exported_model)

        if export_mode == torch.onnx.TrainingMode.TRAINING:
            if enable_custom_autograd_function:
                from ._custom_autograd_function_exporter import post_process_enabling_autograd_function

                post_processed_model = post_process_enabling_autograd_function(post_processed_model)

            if enable_zero_stage3_support:
                from ._zero_stage3_compatibility import post_processing_enable_zero_stage3_compat

                post_processed_model = post_processing_enable_zero_stage3_compat(
                    post_processed_model,
                    stage3_param_handle._zero_stage3_param_map,
                    [name for name, _ in flatten_module.named_parameters()],
                )

                # Cannot append pull weight trigger name to input names as following, otherwise, the later check (
                # https://github.com/microsoft/onnxruntime/blob/068300d97eb25e5b52324e7af54a45ed1fa6a4c3/orttraining/orttraining/python/training/ortmodule/_training_manager.py#L466C18-L466C18)
                # find input info mismatch, will re-initialize the graph builder.
                # self._input_info.require_grad_names.append(STAGE3_PULL_WEIGHT_TRIGGER_NAME)

        post_export_processed_model_info = PostExportProcessedModelInfo(
            flatten_module,
            exported_model_info.onnx_graph_input_names_user_defined,
            exported_model_info.onnx_graph_input_names_require_grad_user_defined,
            exported_model_info.onnx_graph_input_names,
            exported_model_info.onnx_graph_input_names_require_grad,
            model_info_for_export.onnx_graph_input_dynamic_axes_map,
            exported_model_info.module_forward_output_schema,
            post_processed_model,
            model_info_for_export.data_accessor,
        )

        logger.info(
            f"_post_export_process completed, post-export processed graph infos: {post_export_processed_model_info}"
        )

        return post_export_processed_model_info

    # @_logger.TrackTime(_logger.ORTModuleInitPhase.EXPORT)
    # @_logger.SuppressLogs(_logger.ORTModuleInitPhase.EXPORT, is_ort_filter=False)
    @staticmethod
    def _export_model(
        flattened_module: torch.nn.Module,
        model_info_for_export: _io.ModelInfoForExport,
        flatten_module_inputs: Sequence[ORTModelInputOutputType],
        run_symbolic_shape_infer: bool,
        deepcopy_before_model_export: bool,
        device: torch.device,
        ortmodule_cache_dir: str,
        enable_custom_autograd_function: bool,
        enable_zero_stage3_support: bool,
        onnx_opset_version: int,
        torch_exporter_verbose_log: bool,
        stage3_param_handle: type,
        logger: logging.Logger,
    ) -> tuple[onnx.ModelProto, ORTModelInputOutputSchemaType, list[str], list[str]]:
        # Record random states here and restore later in case any of them gets changed during the export,
        # e.g., some sympy functions in symbolic_shape_infer will change Python's random state.
        random_states = _utils.get_random_states()

        from onnxruntime.training.utils.hooks._subscriber_manager import no_increase_global_step

        with no_increase_global_step():
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

        if run_symbolic_shape_infer:
            exported_model = SymbolicShapeInference.infer_shapes(
                exported_model, auto_merge=True, guess_output_rank=True
            )

        # Restore the recorded random states
        _utils.set_random_states(random_states)

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
            dynamic_axes,
            module_output_schema,
        ) = _io.parse_outputs_for_onnx_export_and_extract_schema(
            flattened_module, flatten_module_inputs, logger, need_deep_copy
        )

        # Combine the dynamic axes from inputs and outputs
        dynamic_axes.update(model_info_for_export.onnx_graph_input_dynamic_axes_map)

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
            with torch.no_grad(), stage3_export_context(
                enable_zero_stage3_support, stage3_param_handle, flattened_module
            ):
                required_export_kwargs = {
                    "input_names": model_info_for_export.onnx_graph_input_names,  # did not contains paramerter as its input yet
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
