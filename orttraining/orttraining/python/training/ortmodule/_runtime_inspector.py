# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import tempfile
from enum import IntEnum
from logging import Logger
from typing import Dict, List, Optional, Tuple, Union

import onnx
import torch
from sympy import Symbol, simplify
from sympy.parsing.sympy_parser import parse_expr

from onnxruntime.training.utils import PTable, log_memory_usage

from ._execution_agent import TrainingAgent
from .options import _MemoryOptimizationLevel, _RuntimeOptions


class Phase(IntEnum):
    INVALID = -1
    PRE_FORWARD = 0
    POST_FORWARD = 1
    PRE_BACKWARD = 2  # not applicable for inference
    POST_BACKWARD = 3  # not applicable for inference


def _convert_phase_to_string(phase: Phase) -> str:
    if phase == Phase.PRE_FORWARD:
        return "pre_forward"
    elif phase == Phase.POST_FORWARD:
        return "post_forward"
    elif phase == Phase.PRE_BACKWARD:
        return "pre_backward"
    elif phase == Phase.POST_BACKWARD:
        return "post_backward"
    else:
        return "invalid"


class RuntimeInspector:
    """
    Runtime inspector for ORTModule.
    """

    def __init__(self, logger: Logger, module: torch.nn.Module, training: bool):
        """Initialize runtime inspector.

        Args:
            logger: Logger.
            module: Torch module.
            training: a boolean indicating whether the module is in training mode.
        """
        self._logger = logger

        self.memory_ob = MemoryObserver(module, self._logger, training)
        self._embedding_module_to_padding_density_map = {}
        self._sceloss_module_to_ignore_density_map = {}


class MemoryOptimizationSummary:
    """Memory optimization summary for a cluster id combination."""

    def __init__(self, saving_str="", simplified_saving_expr=None, evaluated_saving=None, freq=0):
        self.raw_symbolic_saving_str = saving_str
        self.simplified_symbolic_saving_expr: Optional[Symbol] = simplified_saving_expr
        self.evaluated_saving: Union[str, int, None] = evaluated_saving
        self.freq = freq


class MemoryObserver:
    """Memory inspector across the training lifetime.

    On different training/inference phases, `inspect_memory` is called to print out the memory usage, including
    current/peak memory usage, current/peak inactive and non-releasable memory.
    """

    NORMALIZER_FACTOR = float(1024 * 1024)
    NORMALIZER_UNIT = "MiB"

    def __init__(self, m: torch.nn.Module, logger: Logger, training: bool):
        """Initialize memory observer.

        Args:
            m: Torch module.
            logger: Logger.
            training: a boolean indicating whether the module is in training mode.
        """
        self._logger = logger
        self._is_enabled = True

        # Memory optimization related.
        self.cluster_id_combination_to_saving_symbolics_map: Dict[str, MemoryOptimizationSummary] = {}
        ## The value is a list of symbolic dim values parsed from the first batch.
        self.symbolic_dim_name_to_value_map: Dict = {}

        ## Used to control only the first batch is used to collect symbolic dim values.
        self.symbolic_dim_collecting_completed = False

        # For per-step memory inspection.
        self._print_memory_stats_by_step = False
        self._current_step = 0
        self._rank = 0
        self._world_size = 1
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()

        self._rank_info = f"[{self._rank}/{self._world_size}]"
        self._pre_phase = Phase.INVALID

        # Cannot infer it is for training or inferencing purpose from module.training,
        # because it probabbly is not set correctly when this happens.
        self._last_phase = Phase.POST_BACKWARD if training else Phase.POST_FORWARD

        self._is_first_inspect = True

        self._m = m

        self._json_file_for_layerwise_recompute = None

    def is_enabled(self) -> bool:
        """Check if memory inspector is enabled."""
        return self._is_enabled

    def enable_memory_stats_by_step(self, print_memory_stats_by_step: bool):
        # For per-step memory inspection.
        self._print_memory_stats_by_step = print_memory_stats_by_step

    def collect_symbolic_dim_values(
        self,
        onnx_input_name_to_dynamic_axes_map: Dict[str, Dict[int, str]],
        onnx_input_to_value_map: Dict[str, torch.Tensor],
    ):
        """Collect symbolic dim values."""
        for input_name, dynamic_axes in onnx_input_name_to_dynamic_axes_map.items():
            if input_name in onnx_input_to_value_map:
                for dim_idx, dim_name in dynamic_axes.items():
                    self.symbolic_dim_name_to_value_map[Symbol(dim_name)] = onnx_input_to_value_map[input_name].size()[
                        dim_idx
                    ]

    def find_memory_optimization_opportunity(self, execution_agent: TrainingAgent, runtime_options: _RuntimeOptions):
        """Find memory optimization opportunity.

        Args:
            execution_agent: TrainingAgent.
            runtime_options: Runtime options.
        """

        recompute_probe_config = runtime_options.recompute_probe_config
        memory_optimizer_config_file_path = runtime_options.memory_optimizer_config_file_path

        # If the memory optimization level is aggressive, we will first collect all
        # recompute subgraph by passing empty memory_optimizer_config_file_path to get_serialized_ortmodule_memory_stat.
        if runtime_options.memory_optimization_level in [
            _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE,
            _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE_WITH_COMPROMISE,
        ]:
            memory_optimizer_config_file_path = ""

        (
            _,
            memory_optimization_saving_symbolics,
        ) = execution_agent.get_serialized_ortmodule_memory_stat(
            memory_optimizer_config_file_path, recompute_probe_config, False
        )

        cluster_id_to_saving_symbol_map: Dict[str, MemoryOptimizationSummary] = {}
        for cluster_id, memory_saving_stat in memory_optimization_saving_symbolics.items():
            memory_saving_symbolic = memory_saving_stat[0]
            freq = memory_saving_stat[1]
            expr = parse_expr(memory_saving_symbolic)
            simplified_expr = simplify(expr)
            r = simplified_expr.evalf(subs=self.symbolic_dim_name_to_value_map)
            evaluated_saving = None
            if r.is_number:
                evaluated_saving = float(r)
            else:
                evaluated_saving = r

            cluster_id_to_saving_symbol_map[cluster_id] = MemoryOptimizationSummary(
                memory_saving_symbolic, simplified_expr, evaluated_saving, freq
            )

        # Sorted by evaluated_saving if it is a float
        sorted_list = sorted(
            cluster_id_to_saving_symbol_map.items(),
            key=lambda x: x[1].evaluated_saving if isinstance(x[1].evaluated_saving, float) else 0,
            reverse=True,
        )

        for cluster_id, values in sorted_list:
            self.cluster_id_combination_to_saving_symbolics_map[cluster_id] = values

        # For aggressive memory optimization, we update the memory_optimizer_config_file_path using all.
        if runtime_options.memory_optimization_level in [
            _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE,
            _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE_WITH_COMPROMISE,
        ]:
            apply_config = []

            for cluster_id in self.cluster_id_combination_to_saving_symbolics_map:
                plans = cluster_id.split(",")
                recompute_configs = []
                for plan in plans:
                    config_values = plan.split(":")
                    opt_type = int(config_values[1])
                    if (
                        runtime_options.memory_optimization_level
                        == _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE
                        and opt_type == _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE
                    ):
                        recompute_configs.append(plan)
                    elif (
                        runtime_options.memory_optimization_level
                        == _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE_WITH_COMPROMISE
                        and opt_type
                        in [
                            _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE,
                            _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE_WITH_COMPROMISE,
                        ]
                    ):
                        recompute_configs.append(plan)

                apply_config.append(",".join(recompute_configs))

            self._json_file_for_layerwise_recompute = tempfile.NamedTemporaryFile(mode="w+")  # noqa: SIM115
            json.dump(apply_config, self._json_file_for_layerwise_recompute)
            self._json_file_for_layerwise_recompute.flush()
            runtime_options.memory_optimizer_config_file_path = self._json_file_for_layerwise_recompute.name

    def inspect_memory(self, cur_phase: Phase):
        """Inspect memory usage and print statistics.

        Args:
            phase: Phase to inspect.
        """

        if not torch.cuda.is_available() or not self._print_memory_stats_by_step:
            return

        if self._is_first_inspect:
            # Clean the memory cache and memory stats before the first time run forward pass, FOR EVERY RANK.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self._is_first_inspect = False

        if self._rank != 0:
            return

        if cur_phase < Phase.PRE_FORWARD or (cur_phase > Phase.POST_BACKWARD):
            raise RuntimeError(f"Invalid phase detected: {cur_phase}, last_phase: {self._last_phase}")

        if (cur_phase - self._pre_phase) != 1:
            raise RuntimeError(f"Invalid phase transition detected: {self._pre_phase} -> {cur_phase}")

        # For the 10+ steps, only print when it is power of 2.
        need_print = self._current_step < 10 or (self._current_step & (self._current_step - 1) == 0)

        if need_print:
            log_memory_usage(
                _convert_phase_to_string(cur_phase),
                rank_0_only=True,
                step_info=f"step {self._current_step}",
                logger=self._logger,
                module=self._m,
            )

        if cur_phase == self._last_phase:
            self._increase_step()
            self._pre_phase = Phase.INVALID
            return

        self._pre_phase = cur_phase

    def _increase_step(self):
        self._current_step += 1

    def display_memory_optimization_plans(
        self, memory_optimizer_config_file_path, details=False
    ) -> Tuple[List[str], PTable]:
        mem_plan_count = len(self.cluster_id_combination_to_saving_symbolics_map)

        if mem_plan_count > 0:
            mem_tbl = PTable()
            if details:
                mem_tbl.add_row(["", "", "", "", "Configs", "Freq", "Max Saving(Bytes)", "Saving Symbolic(Bytes)"])

            index = 1

            def _get_user_config_without_freq(configs: str):
                if len(configs) == 0:
                    return []
                config_list = configs.split(",")
                configs_with_out_freq = []
                for config in config_list:
                    config_values = config.split(":")
                    freq = int(config_values[2])
                    if freq == 0:
                        continue
                    configs_with_out_freq.append(config_values[0] + ":" + config_values[1])

                return configs_with_out_freq

            user_configs_with_out_freq = []
            if memory_optimizer_config_file_path:
                with open(memory_optimizer_config_file_path) as conf:
                    data = json.load(conf)
                    for user_specified_plan in data:
                        user_configs_with_out_freq.extend(_get_user_config_without_freq(user_specified_plan))

            for (
                cluster_id,
                saving_symbolic,
            ) in self.cluster_id_combination_to_saving_symbolics_map.items():
                saving_bytes = saving_symbolic.evaluated_saving
                if isinstance(saving_bytes, float):
                    saving_bytes = f"{saving_bytes:,.0f}"

                cluster_ids_without_freq = _get_user_config_without_freq(cluster_id)

                mem_tbl.add_row(
                    [
                        f" - Plan {index}",
                        ":",
                        (
                            "ON"
                            if all(cluster_id in user_configs_with_out_freq for cluster_id in cluster_ids_without_freq)
                            else "OFF"
                        ),
                        ":",
                        cluster_id,
                        saving_symbolic.freq if details else "",
                        saving_bytes if details else "",
                        saving_symbolic.simplified_symbolic_saving_expr if details else "",
                    ]
                )

                index += 1

            notes = []
            if details:
                notes.append(
                    "Use ORTMODULE_MEMORY_OPT_LEVEL=1 or 2 to enable all recomputable subgraphs per transformer layer."
                )
                saving_recommendation = (
                    "Or use comma as a delimiter to selectively enable multiple memory optimization plans:\n"
                )
                saving_recommendation += "  export ORTMODULE_MEMORY_OPT_CONFIG=<config.json>"

                notes.append(saving_recommendation)

                saving_recommendation = "Memory saving is calculated based on the 1st batch symbolic dim values:\n"
                for dim_param, dim_value in self.symbolic_dim_name_to_value_map.items():
                    saving_recommendation += f"  {dim_param}={dim_value},"
                notes.append(saving_recommendation)

            return notes, mem_tbl

        return [], None


class FlagAndPrintDensity(torch.autograd.Function):
    """
    FlagAndPrintDensity is a PyTorch autograd function that print input density for embedding or label.
    It is also used as a flag to tell the GraphTransformer of PaddingElimination and InsertGatherBeforeSceLoss
    to modify the graph to eliminate the padding.
    """

    @staticmethod
    def forward(ctx, input, padding_value, type_name):
        valid_token = torch.count_nonzero(input - padding_value)
        total_token = input.numel()
        density = float(valid_token) / float(total_token) * 100
        print(type_name + " tensor density: ", density)
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output

    @staticmethod
    def infer_shape(
        node: onnx.NodeProto,
        tensor_input_shapes: List[Optional[List[Union[int, str]]]],
        tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
    ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
        return tensor_input_shapes, tensor_input_dtypes

    @staticmethod
    def alias_input(node_proto_str: str):
        fw_alias_map = [0]
        bw_alias_map = [0, -1, -1]
        return fw_alias_map, bw_alias_map
