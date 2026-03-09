# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Optional

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    add_telemetry_options,
    get_input_model_config,
)
from olive.common.utils import set_nested_dict_value
from olive.constants import Precision, precision_bits_from_precision
from olive.hardware.constants import ExecutionProvider
from olive.telemetry import action


class OptimizeCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "optimize",
            help="Optimize the input model with comprehensive pass scheduling",
        )

        # Model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            default_output_path="optimized-model",
        )

        # Execution provider options
        sub_parser.add_argument(
            "--provider",
            type=str,
            default=ExecutionProvider.CPUExecutionProvider,
            choices=[
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "QNNExecutionProvider",
                "VitisAIExecutionProvider",
                "OpenVINOExecutionProvider",
                "WebGpuExecutionProvider",
                "NvTensorRTRTXExecutionProvider",
            ],
            help="Execution provider (EP) to use for optimization.",
        )

        # Device options
        sub_parser.add_argument(
            "--device",
            type=str,
            default=None,
            choices=["cpu", "gpu", "npu"],
            help="Target device for optimization.",
        )

        # Precision options
        sub_parser.add_argument(
            "--precision",
            type=str,
            default=Precision.FP32,
            choices=[
                Precision.INT4,
                Precision.INT8,
                Precision.INT16,
                Precision.INT32,
                Precision.UINT4,
                Precision.UINT8,
                Precision.UINT16,
                Precision.UINT32,
                Precision.FP16,
                Precision.FP32,
                Precision.BF16,
            ],
            help="Target precision for optimization.",
        )

        # Optional activation precision
        sub_parser.add_argument(
            "--act_precision",
            type=str,
            choices=[Precision.INT8, Precision.UINT8, Precision.INT16, Precision.UINT16],
            help="Activation precision for quantization (optional).",
        )

        # Model splitting options
        sub_parser.add_argument(
            "--num_split",
            type=int,
            help="Number of splits for model splitting (optional).",
        )

        sub_parser.add_argument(
            "--memory",
            type=int,
            help="Available device memory in MB (optional).",
        )

        # Exporter options
        sub_parser.add_argument(
            "--exporter",
            type=str,
            choices=["model_builder", "dynamo_exporter", "torchscript_exporter", "optimum_exporter"],
            help="Exporter to use for model conversion (optional).",
        )

        # Dynamic shape options
        sub_parser.add_argument(
            "--dim_param",
            type=str,
            help="Dynamic parameter names for dynamic to fixed shape conversion (optional).",
        )

        sub_parser.add_argument(
            "--dim_value",
            type=str,
            help="Fixed dimension values for dynamic to fixed shape conversion (optional).",
        )

        # QDQ format option
        sub_parser.add_argument(
            "--use_qdq_format",
            action="store_true",
            help="Use QDQ format for quantization instead of QOperator format.",
        )

        # Graph surgeries option
        sub_parser.add_argument(
            "--surgeries",
            type=str,
            nargs="*",
            help="List of graph surgeries to apply (optional).",
        )

        # Block size option
        sub_parser.add_argument(
            "--block_size",
            type=int,
            help="Block size for quantization. Use -1 for per-channel quantization (optional).",
        )

        # Modality option
        sub_parser.add_argument(
            "--modality",
            type=str,
            default="text",
            choices=["text"],
            help="Model modality for optimization. Only 'text' is currently supported.",
        )

        # QDQ format option
        sub_parser.add_argument(
            "--enable_aot",
            action="store_true",
            help="Enable Ahead-of-Time (AOT) compilation.",
        )

        # QNN environment path option
        sub_parser.add_argument(
            "--qnn_env_path",
            type=str,
            help="Path to QNN environment directory (required when using AOT with QNN).",
        )

        # Extra options for model builder
        sub_parser.add_argument(
            "--extra_mb_options",
            type=str,
            required=False,
            help="Extra key-value pairs options to pass to the model builder. e.g., 'int4_is_symmetric=true,int4_op_types_to_quantize=MatMul/Gemm'.",
        )

        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=OptimizeCommand)

    def __init__(self, parser: ArgumentParser, args: Namespace, unknown_args: Optional[list] = None):
        super().__init__(parser, args, unknown_args)
        self.need_wikitest_data_config = False
        self.is_hf_model = False  # will be set in _get_run_config

        # Pass enabled flags
        self.enable_quarot = False
        self.enable_gptq = False
        self.enable_capture_split_info = False
        self.enable_model_builder = False
        self.enable_onnx_conversion = False
        self.enable_optimum_openvino_conversion = False
        self.enable_dynamic_to_fixed_shape = False
        self.enable_vitis_ai_preprocess = False
        self.enable_onnx_io_datatype_converter = False
        self.enable_openvino_io_update = False
        self.enable_onnx_peephole_optimizer = False
        self.enable_matmul_nbits_to_qdq = False
        self.enable_graph_surgeries = False
        self.enable_onnx_blockwise_rtn_quantization = False
        self.enable_onnx_float_to_float16 = False
        self.enable_onnx_static_quantization = False
        self.enable_ort_transformers_optimization = False
        self.enable_split_model = False
        self.enable_static_llm = False
        self.enable_vitis_ai_add_metadata = False
        self.enable_ep_context_binary_generator = False
        self.enable_compose_onnx_models = False
        self.enable_openvino_encapsulation = False

    @action
    def run(self):
        return self._run_workflow()

    def _get_run_config(self, tempdir: str) -> dict[str, Any]:
        config = deepcopy(TEMPLATE)

        # Handle arguments
        self._validate_arguments()

        # Set input model configuration
        config["input_model"] = get_input_model_config(self.args)
        self.is_hf_model = config["input_model"]["type"].lower() == "hfmodel"

        # Build the pass list based on conditions
        passes_config = self._build_passes_config()
        config["passes"] = passes_config

        # Set data config
        self._add_data_config(config)

        # Set system configuration
        self._update_system_config(config)

        # Apply customizations
        to_replace = [
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]
        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)

        return config

    def _validate_arguments(self):
        if self.args.exporter is None and self.args.modality == "text":
            self.args.exporter = "model_builder"

        if self.args.modality not in ["text"]:
            raise ValueError(f"Unsupported modality: {self.args.modality}. Only 'text' is supported for optimization.")

        if self.args.provider == ExecutionProvider.CPUExecutionProvider and self.args.device in ["gpu", "npu"]:
            raise ValueError(
                f"Invalid combination of provider {self.args.provider} and device {self.args.device}. "
                "Please use a compatible provider for the specified device."
            )

        if self.args.provider == ExecutionProvider.CUDAExecutionProvider and self.args.device in ["cpu", "npu"]:
            raise ValueError(
                f"Invalid combination of provider {self.args.provider} and device {self.args.device}. "
                "Please use a compatible provider for the specified device."
            )

        if self.args.provider == ExecutionProvider.NvTensorRTRTXExecutionProvider and self.args.device in [
            "cpu",
            "npu",
        ]:
            raise ValueError(
                f"Invalid combination of provider {self.args.provider} and device {self.args.device}. "
                "Please use a compatible provider for the specified device."
            )

        if self.args.enable_aot and self.args.provider != ExecutionProvider.QNNExecutionProvider:
            raise ValueError("Ahead-of-Time (AOT) compilation is only supported with QNNExecutionProvider.")

        if self.args.enable_aot and self.args.qnn_env_path is None:
            raise ValueError("QNN environment path (--qnn_env_path) is required when using AOT compilation.")

        if self.args.use_qdq_format and self.args.provider == ExecutionProvider.OpenVINOExecutionProvider:
            raise ValueError("QDQ format is not supported with OpenVINOExecutionProvider.")

    def _update_system_config(self, config: dict[str, Any]):
        """Update system configuration based on provider and device."""
        provider = ExecutionProvider(self.args.provider)

        if provider == ExecutionProvider.QNNExecutionProvider and self.args.enable_aot:
            config["systems"]["qnn_system"] = {
                "type": "PythonEnvironment",
                "python_environment_path": self.args.qnn_env_path,
                "accelerators": [{"execution_providers": [provider.value]}],
            }
            config["target"] = "qnn_system"

    def _add_data_config(self, config: dict[str, Any]):
        config["data_configs"] = WIKITEXT2_DATA_CONFIG_TEMPLATE if self.need_wikitest_data_config else []

    def _build_passes_config(self) -> dict[str, Any]:
        passes_config = OrderedDict()

        self.enable_quarot = self._enable_quarot_pass()
        if self.enable_quarot:
            passes_config["quarot"] = self._get_quarot_pass_config()

        self.enable_gptq = self._enable_gptq_pass()
        if self.enable_gptq:
            passes_config["gptq"] = self._get_gptq_pass_config()

        self.enable_capture_split_info = self._enable_capture_split_info_pass()
        if self.enable_capture_split_info:
            passes_config["capture_split_info"] = self._get_capture_split_info_pass_config()

        self.enable_model_builder = self._enable_model_builder_pass()
        if self.enable_model_builder:
            passes_config["model_builder"] = self._get_model_builder_pass_config()

        self.enable_onnx_conversion = self._enable_onnx_conversion_pass()
        if self.enable_onnx_conversion:
            passes_config["onnx_conversion"] = self._get_onnx_conversion_pass_config()

        self.enable_optimum_openvino_conversion = self._enable_optimum_openvino_conversion_pass()
        if self.enable_optimum_openvino_conversion:
            passes_config["optimum_openvino_conversion"] = self._get_optimum_openvino_conversion_pass_config()

        self.enable_dynamic_to_fixed_shape = self._enable_dynamic_to_fixed_shape_pass()
        if self.enable_dynamic_to_fixed_shape:
            passes_config["dynamic_to_fixed_shape"] = self._get_dynamic_to_fixed_shape_pass_config()

        self.enable_onnx_io_datatype_converter = self._enable_onnx_io_datatype_converter_pass()
        if self.enable_onnx_io_datatype_converter:
            passes_config["onnx_io_datatype_converter"] = self._get_onnx_io_datatype_converter_pass_config()

        self.enable_openvino_io_update = self._enable_openvino_io_update_pass()
        if self.enable_openvino_io_update:
            passes_config["openvino_io_update"] = self._get_openvino_io_update_pass_config()

        self.enable_onnx_peephole_optimizer = self._enable_onnx_peephole_optimizer_pass()
        if self.enable_onnx_peephole_optimizer:
            passes_config["onnx_peephole_optimizer"] = self._get_onnx_peephole_optimizer_pass_config()

        self.enable_ort_transformers_optimization = self._enable_ort_transformers_optimization_pass()
        if self.enable_ort_transformers_optimization:
            passes_config["ort_transformers_optimization"] = self._get_ort_transformers_optimization_pass_config()

        self.enable_matmul_nbits_to_qdq = self._enable_matmul_nbits_to_qdq_pass(passes_config)
        if self.enable_matmul_nbits_to_qdq:
            passes_config["matmul_nbits_to_qdq"] = self._get_matmul_nbits_to_qdq_pass_config()

        self.enable_graph_surgeries = self._enable_graph_surgeries_pass()
        if self.enable_graph_surgeries:
            passes_config["graph_surgeries"] = self._get_graph_surgeries_pass_config()

        self.enable_onnx_blockwise_rtn_quantization = self._enable_onnx_blockwise_rtn_quantization_pass()
        if self.enable_onnx_blockwise_rtn_quantization:
            passes_config["onnx_blockwise_rtn_quantization"] = self._get_onnx_blockwise_rtn_quantization_pass_config()

        self.enable_onnx_float_to_float16 = self._enable_onnx_float_to_float16_pass()
        if self.enable_onnx_float_to_float16:
            passes_config["onnx_float_to_float16"] = self._get_onnx_float_to_float16_pass_config()

        self.enable_onnx_static_quantization = self._enable_onnx_static_quantization_pass()
        if self.enable_onnx_static_quantization:
            passes_config["onnx_static_quantization"] = self._get_onnx_static_quantization_pass_config()

        self.enable_vitis_ai_add_metadata = self._enable_vitis_ai_add_metadata_pass()
        if self.enable_vitis_ai_add_metadata:
            passes_config["vitis_ai_add_metadata"] = self._get_vitis_ai_add_metadata_pass_config()

        self.enable_split_model = self._enable_split_model_pass()
        if self.enable_split_model:
            passes_config["split_model"] = self._get_split_model_pass_config()

        self.enable_static_llm = self._enable_static_llm_pass()
        if self.enable_static_llm:
            passes_config["static_llm"] = self._get_static_llm_pass_config()

        self.enable_ep_context_binary_generator = self._enable_ep_context_binary_generator_pass()
        if self.enable_ep_context_binary_generator:
            passes_config["ep_context_binary_generator"] = self._get_ep_context_binary_generator_pass_config()

        self.enable_compose_onnx_models = self._enable_compose_onnx_models_pass()
        if self.enable_compose_onnx_models:
            passes_config["compose_onnx_models"] = self._get_compose_onnx_models_pass_config()

        self.enable_openvino_encapsulation = self._enable_openvino_encapsulation_pass()
        if self.enable_openvino_encapsulation:
            passes_config["openvino_encapsulation"] = self._get_openvino_encapsulation_pass_config()

        return passes_config

    def _is_pt_quantized_precision(self, precision: Precision) -> bool:
        # Helper function to check if precision is quantized.
        return precision in [Precision.INT4, Precision.UINT4]

    def _enable_quarot_pass(self) -> bool:
        """Return true if condition to add QuaRot pass is met."""
        provider = ExecutionProvider(self.args.provider)
        precision = Precision(self.args.precision)
        return (
            self._is_pt_quantized_precision(precision)
            and self.is_hf_model
            and provider in [ExecutionProvider.QNNExecutionProvider, ExecutionProvider.VitisAIExecutionProvider]
        )

    def _get_quarot_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for QuaRot pass."""
        return {"type": "QuaRot"}

    def _enable_gptq_pass(self) -> bool:
        """Return true if condition to add Gptq pass is met."""
        provider = ExecutionProvider(self.args.provider)
        precision = Precision(self.args.precision)
        return (
            self.is_hf_model
            and self._is_pt_quantized_precision(precision)
            and provider != ExecutionProvider.OpenVINOExecutionProvider
        )

    def _get_gptq_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for Gptq pass."""
        precision = Precision(self.args.precision)
        precision_bits = precision_bits_from_precision(precision)
        bits = precision_bits.value if precision_bits else 32
        gptq_config = {"type": "Gptq", "bits": bits, "sym": precision == Precision.INT4}
        if self.args.block_size is not None:
            if self.args.block_size == -1:
                # For per-channel quantization in GPTQ, use a special value or handle differently
                # Based on the INC quantization pattern, -1 typically means per-channel
                gptq_config["group_size"] = -1
            else:
                gptq_config["group_size"] = self.args.block_size
        return gptq_config

    def _enable_capture_split_info_pass(self) -> bool:
        """Return true if condition to add CaptureSplitInfo pass is met."""
        return self.is_hf_model and (self.args.num_split is not None or self.args.memory is not None)

    def _get_capture_split_info_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for CaptureSplitInfo pass."""
        config = {"type": "CaptureSplitInfo"}
        config["unique_embeds_lm_head_splits"] = True
        if self.args.num_split is not None:
            config["num_splits"] = self.args.num_split
        if self.args.memory is not None:
            config["memory"] = self.args.memory
        return config

    def _enable_model_builder_pass(self) -> bool:
        """Return true if condition to add ModelBuilder pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return (
            self.is_hf_model
            and provider != ExecutionProvider.OpenVINOExecutionProvider
            and self.args.exporter == "model_builder"
        )

    def _get_model_builder_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for ModelBuilder pass."""
        precision = Precision(self.args.precision)
        config = {"type": "ModelBuilder", "precision": precision.value}
        if precision.value == Precision.INT4:
            # Use provided block_size if available, otherwise default to 32
            block_size_value = self.args.block_size if self.args.block_size is not None else 32
            # For ModelBuilder, -1 block_size should use a reasonable default since it doesn't support per-channel
            if block_size_value == -1:
                block_size_value = 32
            # Ensure block_size is valid for ModelBuilder (16, 32, 64, 128, 256)
            valid_block_sizes = [16, 32, 64, 128, 256]
            if block_size_value not in valid_block_sizes:
                # Find the closest valid block size
                block_size_value = min(valid_block_sizes, key=lambda x: abs(x - block_size_value))
            config["int4_block_size"] = block_size_value
            config["int4_accuracy_level"] = 4
            config["int4_op_types_to_quantize"] = ["MatMul", "Gather"]

        extra_options = {}
        if self.args.extra_mb_options:
            extra_options = BaseOliveCLICommand._parse_extra_options(self.args.extra_mb_options.split(","))
        config["extra_options"] = extra_options

        return config

    def _enable_onnx_conversion_pass(self) -> bool:
        """Return true if condition to add OnnxConversion pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return (
            self.is_hf_model
            and provider != ExecutionProvider.OpenVINOExecutionProvider
            and self.args.exporter in ["dynamo_exporter", "torchscript_exporter"]
        )

    def _get_onnx_conversion_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OnnxConversion pass."""
        return {"type": "OnnxConversion", "use_dynamo_exporter": self.args.exporter == "dynamo_exporter"}

    def _enable_optimum_openvino_conversion_pass(self) -> bool:
        """Return true if condition to add OptimumOpenvinoConversion pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return self.is_hf_model and provider == ExecutionProvider.OpenVINOExecutionProvider

    def _get_optimum_openvino_conversion_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OptimumOpenvinoConversion pass."""
        return {
            "type": "OpenVINOOptimumConversion",
            "extra_args": {"device": self.args.device},
            "ov_quant_config": {
                "task": "text-generation-with-past",
                "weight_format": Precision(self.args.precision).value,
                "group_size": 128,
                "ratio": 1,
            },
        }

    def _enable_dynamic_to_fixed_shape_pass(self) -> bool:
        """Return true if condition to add DynamicToFixedShape pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return (
            (
                provider in [ExecutionProvider.QNNExecutionProvider, ExecutionProvider.VitisAIExecutionProvider]
                or self.args.device == "npu"
            )
            and self.args.dim_param is not None
            and self.args.dim_value is not None
        )

    def _get_dynamic_to_fixed_shape_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for DynamicToFixedShape pass."""
        return {
            "type": "DynamicToFixedShape",
            "dim_param": [item.strip() for item in self.args.dim_param.split(",")],
            "dim_value": [int(item.strip()) for item in self.args.dim_value.split(",")],
        }

    def _enable_openvino_io_update_pass(self) -> bool:
        """Return true if condition to add OpenVINOIoUpdate pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return provider == ExecutionProvider.OpenVINOExecutionProvider and self.is_hf_model

    def _get_openvino_io_update_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OpenVINOIoUpdate pass."""
        return {"type": "OpenVINOIoUpdate", "static": False, "reuse_cache": True}

    def _enable_onnx_peephole_optimizer_pass(self) -> bool:
        """Return true if condition to add OnnxPeepholeOptimizer pass is met."""
        return self.args.exporter != "model_builder"

    def _get_onnx_peephole_optimizer_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OnnxPeepholeOptimizer pass."""
        return {"type": "OnnxPeepholeOptimizer"}

    def _enable_ort_transformers_optimization_pass(self) -> bool:
        """Return true if condition to add OrtTransformersOptimization pass is met."""
        provider = ExecutionProvider(self.args.provider)
        # Do not enable OrtTransformersOptimization when using NVTensorRtRTX EP
        if provider == ExecutionProvider.NvTensorRTRTXExecutionProvider:
            return False
        return self.args.exporter in ["torchscript_exporter", "dynamo_exporter"]

    def _get_ort_transformers_optimization_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OrtTransformersOptimization pass."""
        return {"type": "OrtTransformersOptimization"}

    def _enable_matmul_nbits_to_qdq_pass(self, passes_config: dict[str, Any]) -> bool:
        """Return true if condition to add MatMulNBitsToQDQ pass is met."""
        return self.is_hf_model and "gptq" in passes_config and self.args.use_qdq_format

    def _get_matmul_nbits_to_qdq_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for MatMulNBitsToQDQ pass."""
        precision = Precision(self.args.precision)
        config = {
            "type": "MatMulNBitsToQDQ",
            "add_zero_point": "true",
            "save_as_external_data": "true",
        }
        config["nodes_to_exclude"] = ["/lm_head/MatMul_Q4"]
        if precision.value == Precision.INT4:
            config["use_int4"] = "true"
        return config

    def _enable_graph_surgeries_pass(self) -> bool:
        """Return true if condition to add GraphSurgeries pass is met."""
        return self.args.surgeries is not None

    def _get_graph_surgeries_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for GraphSurgeries pass."""
        surgeries_list = [{"surgeon": item} for item in self.args.surgeries[0].split(",")]
        return {
            "type": "GraphSurgeries",
            "surgeries": surgeries_list,
            "save_as_external_data": "true",
        }

    def _enable_onnx_blockwise_rtn_quantization_pass(self) -> bool:
        """Return true if condition to add OnnxBlockWiseRtnQuantization pass is met."""
        precision = Precision(self.args.precision)
        return not self.is_hf_model and precision == Precision.INT4

    def _get_onnx_blockwise_rtn_quantization_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OnnxBlockWiseRtnQuantization pass."""
        config = {"type": "OnnxBlockWiseRtnQuantization"}
        if self.args.block_size is not None:
            if self.args.block_size == -1:
                # For per-channel quantization, we can use axis=0 and set block_size to a large value
                # or let the pass handle per-channel internally
                config["axis"] = 0
                # Some implementations use block_size = -1 to indicate per-channel
                config["block_size"] = -1
            else:
                config["block_size"] = self.args.block_size
        return config

    def _enable_onnx_float_to_float16_pass(self) -> bool:
        """Return true if condition to add OnnxFloatToFloat16 pass is met."""
        precision = Precision(self.args.precision)
        return precision == Precision.FP16

    def _get_onnx_float_to_float16_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OnnxFloatToFloat16 pass."""
        return {"type": "OnnxFloatToFloat16"}

    def _enable_onnx_static_quantization_pass(self) -> bool:
        """Return true if condition to add OnnxStaticQuantization pass is met."""
        if self.args.provider == ExecutionProvider.OpenVINOExecutionProvider:
            return False

        precision = Precision(self.args.precision)
        act_precision_check = (
            self.args.act_precision
            in [Precision.INT8.value, Precision.UINT8.value, Precision.INT16.value, Precision.UINT16.value]
            if self.args.act_precision
            else False
        )
        precision_check = (
            precision in [Precision.INT8, Precision.UINT8, Precision.INT16, Precision.UINT16] and not self.enable_gptq
        )
        return precision_check or act_precision_check

    def _get_onnx_static_quantization_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OnnxStaticQuantization pass."""
        precision = Precision(self.args.precision)
        config = {
            "type": "OnnxStaticQuantization",
            "precision": precision.value,
            "calibration_providers": ["CUDAExecutionProvider"],
            "quant_format": "QDQ" if self.args.use_qdq_format else "QOperator",
        }
        if self.args.act_precision:
            config["activation_type"] = self.args.act_precision

        if self.is_hf_model and self.args.modality == "text":
            # these are contrib ops, no need for qdq around them
            config["op_types_to_exclude"] = ["GatherBlockQuantized", "GroupQueryAttention", "MatMulNBits"]
        # Handle block_size parameter
        if self.args.block_size == -1:
            # Use per-channel quantization when block_size is -1
            config["per_channel"] = True

        # Add data_config for text modality
        if self.args.modality == "text":
            self.need_wikitest_data_config = True
            config["data_config"] = "wikitext2_train"

        return config

    def _enable_split_model_pass(self) -> bool:
        """Return true if condition to add SplitModel pass is met."""
        return self.is_hf_model and (self.args.num_split is not None or self.args.memory is not None)

    def _get_split_model_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for SplitModel pass."""
        return {"type": "SplitModel"}

    def _enable_static_llm_pass(self) -> bool:
        """Return true if condition to add StaticLLM pass is met."""
        if self.args.modality != "text":
            return False
        provider = ExecutionProvider(self.args.provider)
        return provider in [ExecutionProvider.QNNExecutionProvider, ExecutionProvider.VitisAIExecutionProvider]

    def _get_static_llm_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for StaticLLM pass."""
        config = {"type": "StaticLLM"}
        if self.args.provider == ExecutionProvider.VitisAIExecutionProvider:
            config["batch_size"] = 1
            config["context_length"] = 64
            config["group_session_options"] = {
                "log_id": "onnxruntime-genai",
                "provider_options": [{"VitisAI": {}}],
                "graph_optimization_level": "ORT_ENABLE_ALL",
            }
        return config

    def _enable_vitis_ai_add_metadata_pass(self) -> bool:
        """Return true if condition to add VitisAIAddMetaData pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return provider == ExecutionProvider.VitisAIExecutionProvider

    def _get_vitis_ai_add_metadata_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for VitisAIAddMetaData pass."""
        config = {
            "type": "VitisAIAddMetaData",
            "config_meta_data_keys": ["architectures", "model_type"],
            "weight_type": Precision(self.args.precision).value,
        }

        act_precision = Precision(self.args.act_precision) if self.args.act_precision else None
        if act_precision:
            config["activation_type"] = act_precision.value

        if self.enable_quarot:
            config["quant_type"] = "quarot"
        elif self.enable_onnx_static_quantization:
            config["quant_type"] = "onnx_static_quantization"
        elif self.enable_gptq:
            config["quant_type"] = "gptq"
        return config

    def _enable_ep_context_binary_generator_pass(self) -> bool:
        """Return true if condition to add EPContextBinaryGenerator pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return self.args.enable_aot and provider == ExecutionProvider.QNNExecutionProvider

    def _get_ep_context_binary_generator_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for EPContextBinaryGenerator pass."""
        config = {
            "type": "EPContextBinaryGenerator",
            "session_options": {"intra_op_num_threads": 2, "inter_op_num_threads": 1},
            "weight_sharing": True,
        }
        config["provider_options"] = {
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
            "soc_model": "60",
        }
        return config

    def _enable_compose_onnx_models_pass(self) -> bool:
        """Return true if condition to add ComposeOnnxModels pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return (
            self.is_hf_model
            and (self.args.enable_aot)
            and (self.args.num_split is not None or self.args.memory is not None)
            and provider == ExecutionProvider.QNNExecutionProvider
        )

    def _get_compose_onnx_models_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for ComposeOnnxModels pass."""
        return {"type": "ComposeOnnxModels"}

    def _enable_openvino_encapsulation_pass(self) -> bool:
        """Return true if condition to add OpenVINOEncapsulation pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return self.is_hf_model and provider == ExecutionProvider.OpenVINOExecutionProvider

    def _get_openvino_encapsulation_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OpenVINOEncapsulation pass."""
        config = {
            "type": "OpenVINOEncapsulation",
            "keep_ov_dynamic_shapes": True,
            "op_version": "2025.1",
            "reuse_cache": True,
        }
        if self.args.device is not None:
            config["target_device"] = self.args.device
        return config

    def _enable_onnx_io_datatype_converter_pass(self) -> bool:
        """Return true if condition to add OnnxIODataTypeConverter pass is met."""
        provider = ExecutionProvider(self.args.provider)
        return provider == ExecutionProvider.WebGpuExecutionProvider

    def _get_onnx_io_datatype_converter_pass_config(self) -> dict[str, Any]:
        """Return pass dictionary for OnnxIODataTypeConverter pass."""
        return {
            "type": "OnnxIODataTypeConverter",
            "name_pattern": "logits",
            "source_dtype": 10,  # FLOAT16
            "target_dtype": 1,  # FLOAT
        }


# Template configuration for the optimize command
TEMPLATE = {
    "input_model": {"type": "HfModel"},
    "passes": OrderedDict(),
    "systems": {},
    "no_artifacts": True,
}

WIKITEXT2_DATA_CONFIG_TEMPLATE = [
    {
        "name": "wikitext2_train",
        "type": "HuggingfaceContainer",
        "load_dataset_config": {"data_name": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1", "split": "train"},
        "pre_process_data_config": {
            "strategy": "line-by-line",
            "add_special_tokens": False,
            "max_samples": 128,
            "max_seq_len": 512,
        },
    }
]
