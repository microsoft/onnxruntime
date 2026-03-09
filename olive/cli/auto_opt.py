# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from olive.cli.base import (
    BaseOliveCLICommand,
    add_accelerator_options,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    add_shared_cache_options,
    add_telemetry_options,
    get_input_model_config,
    update_accelerator_options,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.constants import Precision
from olive.hardware.constants import ExecutionProvider
from olive.package_config import OlivePackageConfig
from olive.telemetry import action


class AutoOptCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "auto-opt",
            help="Automatically optimize the performance of the input model.",
        )

        # Model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            default_output_path="auto-opt-output",
        )

        # add accelerator options
        add_accelerator_options(sub_parser)

        # dataset options
        sub_parser.add_argument(
            "-d",
            "--data_name",
            type=str,
            help="The dataset name.",
        )
        sub_parser.add_argument(
            "--split",
            type=str,
            help="The dataset split to use for evaluation.",
        )
        sub_parser.add_argument(
            "--subset",
            type=str,
            help="The dataset subset to use for evaluation.",
        )
        sub_parser.add_argument(
            "--input_cols",
            type=str,
            nargs="*",
            help="The input columns to use for evaluation.",
        )
        sub_parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size for evaluation.",
        )

        sub_parser.add_argument(
            "--precision",
            type=Precision,
            default=Precision.FP32,
            choices=[v.value for v in Precision],
            help=(
                "The output precision of the optimized model. If not specified, "
                "the default precision is fp32 for cpu and fp16 for gpu"
            ),
        )
        sub_parser.add_argument(
            "--use_dynamo_exporter",
            action="store_true",
            help="Whether to use dynamo_export API to export ONNX model.",
        )
        sub_parser.add_argument(
            "--use_model_builder",
            action="store_true",
            help=(
                "Whether to use model builder pass for optimization, enable only "
                "when the model is supported by model builder"
            ),
        )
        sub_parser.add_argument(
            "--use_qdq_encoding",
            action="store_true",
            help=(
                "Whether to use QDQ encoding for quantized operators instead of ONNXRuntime contrib operators like"
                " MatMulNBits"
            ),
        )

        # DynamicToFixedShape options
        sub_parser.add_argument(
            "--dynamic-to-fixed-shape-dim-param",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Symbolic parameter names to use for dynamic to fixed shape pass. "
                "Required only when using QNNExecutionProvider."
            ),
        )
        sub_parser.add_argument(
            "--dynamic-to-fixed-shape-dim-value",
            type=int,
            nargs="*",
            default=None,
            help=(
                "Symbolic parameter values to use for dynamic to fixed shape pass. "
                "Required only when using QNNExecutionProvider."
            ),
        )

        # Split options
        split_group = sub_parser.add_mutually_exclusive_group(required=False)
        split_group.add_argument(
            "--num-splits",
            type=int,
            help="Number of splits to use for model splitting. Input model must be an HfModel.",
        )
        split_group.add_argument(
            "--cost-model",
            type=str,
            help=(
                "Path to the cost model csv file to use for model splitting. Mutually exclusive with num-splits. Must"
                " be a csv with headers `module,num_params,num_bytes,num_flops` where each row corresponds to the name"
                " or a module (with no children), the number of parameters, the number of bytes, and the number of"
                " FLOPs(batch_size=1, seqlen=1) the module uses when in the desired precision."
            ),
        )

        # MixedPrecisionOverrides options
        sub_parser.add_argument(
            "--mixed-precision-overrides-config",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Dictionary of name to precision. Has to be even number of entreis with even "
                "entries being the keys and odd entries being the values. "
                'Required only when output precision is "fp16" and MixedPrecisionOverrides pass is enabled.'
            ),
        )

        sub_parser.add_argument(
            "--use_ort_genai", action="store_true", help="Use OnnxRuntime generate() API to run the model"
        )

        add_shared_cache_options(sub_parser)
        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=AutoOptCommand)

    @action
    def run(self):
        return self._run_workflow()

    def _get_run_config(self, tempdir) -> dict:
        config = deepcopy(TEMPLATE)
        olive_config = OlivePackageConfig.load_default_config()

        # TODO(anyone): Change add_accelerator_options to have no default device, this can be inferred
        # by create_accelerators
        if (self.args.provider == ExecutionProvider.DmlExecutionProvider) and (self.args.device not in ["gpu", "npu"]):
            # Force the device to gpu for Direct ML provider
            self.args.device = "gpu"
        elif self.args.provider in [ExecutionProvider.QNNExecutionProvider, ExecutionProvider.VitisAIExecutionProvider]:
            self.args.device = "npu"
        elif self.args.provider == ExecutionProvider.CUDAExecutionProvider:
            self.args.device = "gpu"

        # _get_passes_config requires input_model to be set
        config["input_model"] = get_input_model_config(self.args)

        to_replace = [
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]
        to_replace.append(("passes", self._get_passes_config(config, olive_config)))

        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)

        update_accelerator_options(self.args, config)
        update_shared_cache_options(config, self.args)

        return config

    def _get_data_config(self) -> list[dict[str, Any]]:
        if not self.args.data_name:
            return []

        to_replace = [
            (("load_dataset_config", "data_name"), self.args.data_name),
            (("load_dataset_config", "split"), self.args.split),
            (("load_dataset_config", "subset"), self.args.subset),
            (("pre_process_data_config", "input_cols"), self.args.input_cols),
            (("dataloader_config", "batch_size"), self.args.batch_size),
        ]
        data_config = {
            "name": "data_config",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {},
            "pre_process_data_config": {},
            "dataloader_config": {},
        }
        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(data_config, keys, value)

        return [data_config]

    def _get_passes_config(self, config: dict[str, Any], olive_config: OlivePackageConfig) -> dict[str, Any]:
        if self.args.mixed_precision_overrides_config and len(self.args.mixed_precision_overrides_config) % 2 != 0:
            raise ValueError("Even number of entries required for mixed precision overrides config.")

        passes_config: dict[str, Any] = config["passes"]
        mixed_precision_overrides_config = (
            {
                self.args.mixed_precision_overrides_config[i]: self.args.mixed_precision_overrides_config[i + 1]
                for i in range(0, len(self.args.mixed_precision_overrides_config), 2)
            }
            if self.args.mixed_precision_overrides_config
            else None
        )

        to_replace = [
            (("capture_split_info", "num_splits"), self.args.num_splits),
            (("capture_split_info", "cost_model"), self.args.cost_model),
            (("conversion", "use_dynamo_exporter"), self.args.use_dynamo_exporter),
            (("conversion", "save_metadata_for_token_generation"), self.args.use_ort_genai),
            (("bnb4", "precision"), self.args.precision),
            (("dynamic_quant", "precision"), self.args.precision),
            (("model_builder", "precision"), self.args.precision),
            (("genai_config_only", "precision"), self.args.precision),
            # select the float dtype based on the precision, int4 only quantizes matmuls so we still need to set
            # the float precision separately
            (
                ("transformer_optimizer", "float16"),
                self.args.precision == Precision.FP16
                or (
                    self.args.precision == Precision.INT4
                    and self.args.provider != ExecutionProvider.CPUExecutionProvider
                ),
            ),
            (("to_fixed_shape", "dim_param"), self.args.dynamic_to_fixed_shape_dim_param),
            (("to_fixed_shape", "dim_value"), self.args.dynamic_to_fixed_shape_dim_value),
            (("mixed_precision_overrides", "overrides_config"), mixed_precision_overrides_config),
        ]
        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(passes_config, keys, value)

        passes_to_remove = set()
        if config["input_model"]["type"].lower() == "onnxmodel":
            # remove passes that only operate on PyTorch/Hf models
            passes_to_remove.update(["capture_split_info", "conversion", "model_builder", "split_model"])
        else:
            # only one graph capture pass is used
            passes_to_remove.add("conversion" if self.args.use_model_builder else "model_builder")

        # optional split passes
        if self.args.num_splits is None and self.args.cost_model is None:
            passes_to_remove.update(["capture_split_info", "split_model"])
        if self.args.cost_model is not None and self.args.memory is None:
            raise ValueError("memory is required if cost_model is provided.")

        if self.args.provider != ExecutionProvider.QNNExecutionProvider:
            # Use the DynamicToFixedShape pass only for QNNExecutionProvider
            # becase QNN doesn't support dynamic shaped inputs
            passes_to_remove.add("to_fixed_shape")
        else:
            # qnn ep might not supported optimized model
            # will re-enable it if needed in the future
            passes_to_remove.update(["transformer_optimizer", "peephole_optimizer"])

        if self.args.provider not in {ExecutionProvider.JsExecutionProvider, ExecutionProvider.WebGpuExecutionProvider}:
            # JS EP doesn't support fp16 io
            passes_to_remove.add("fp16_to_fp32")

        if self.args.use_model_builder:
            # Don't run optimizers when using model builder
            passes_to_remove.add("transformer_optimizer")
            passes_to_remove.add("peephole_optimizer")
            # model already comes in int4
            passes_to_remove.add("matmul4")

        if self.args.use_model_builder or not self.args.use_ort_genai:
            passes_to_remove.add("genai_config_only")

        if mixed_precision_overrides_config is None:
            # Remove mixed_precision_overrides pass if not required
            passes_to_remove.add("mixed_precision_overrides")

        if not self.args.use_qdq_encoding:
            # Remove QDQ encoding pass if not required
            passes_to_remove.add("mnb_to_qdq")

        # remove passes that are incompatible with the selected precision, provider, or device
        for pass_name in list(passes_config.keys()):
            pass_run_config = passes_config[pass_name]
            pass_module_config = olive_config.get_pass_module_config(pass_run_config["type"])
            if (
                (self.args.precision not in pass_module_config.supported_precisions)
                or (self.args.provider not in pass_module_config.supported_providers)
                or (self.args.device not in pass_module_config.supported_accelerators)
            ):
                passes_to_remove.add(pass_name)

        for pass_name in passes_to_remove:
            del passes_config[pass_name]

        if "to_fixed_shape" in passes_config and not (
            self.args.dynamic_to_fixed_shape_dim_param and self.args.dynamic_to_fixed_shape_dim_value
        ):
            raise ValueError(
                "dynamic-to-fixed-shape-dim-param and dynamic-to-fixed-shape-dim-value are required "
                "when using QNNExecutionProvider."
            )

        # check that there is at least one capture pass for non-onnx models
        if (
            (config["input_model"]["type"].lower() != "onnxmodel")
            and ("conversion" not in passes_config)
            and ("model_builder" not in passes_config)
        ):
            raise ValueError("Cannot export an onnx model with combination of provided options.")

        return passes_config


TEMPLATE = {
    "input_model": {"type": "HfModel"},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "passes": OrderedDict(
        [
            # pytorch related passes
            ("capture_split_info", {"type": "CaptureSplitInfo"}),
            # always convert in float32 since float16 doesn't work for all models
            ("conversion", {"type": "OnnxConversion", "torch_dtype": "float32", "use_dynamo_exporter": False}),
            ("model_builder", {"type": "ModelBuilder", "precision": Precision.FP32}),
            ("genai_config_only", {"type": "ModelBuilder", "precision": Precision.FP32, "metadata_only": True}),
            # model optimization passes
            ("peephole_optimizer", {"type": "OnnxPeepholeOptimizer"}),
            # use transformer optimizer for fp16 conversion too
            # opt_level set to 0 to avoid graph transformations done by onnxruntime inference sessions
            # that are incompatible with later passes. opt_level > 0 is optional and can be done during session creation
            (
                "transformer_optimizer",
                {"type": "OrtTransformersOptimization", "opt_level": 0, "float16": False, "keep_io_types": False},
            ),
            # change io types to fp32
            ("fp16_to_fp32", {"type": "OnnxIODataTypeConverter"}),
            # qnn preparation passes
            ("to_fixed_shape", {"type": "DynamicToFixedShape", "dim_param": None, "dim_value": None}),
            ("qnn_preprocess", {"type": "QNNPreprocess"}),
            ("mixed_precision_overrides", {"type": "MixedPrecisionOverrides", "overrides_config": None}),
            # quantization passes
            ("dynamic_quant", {"type": "OnnxDynamicQuantization", "precision": Precision.INT8}),
            ("matmul4", {"type": "OnnxBlockWiseRtnQuantization"}),
            ("bnb4", {"type": "OnnxBnb4Quantization", "precision": Precision.NF4}),
            # post processing passes
            ("mnb_to_qdq", {"type": "MatMulNBitsToQDQ"}),
            ("split_model", {"type": "SplitModel"}),
            ("extract_adapters", {"type": "ExtractAdapters"}),
        ]
    ),
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
