# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Export a PyTorch model using the onnxruntime-genai package.
# --------------------------------------------------------------------------
import copy
import importlib
import json
import logging
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Union

import onnx
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from packaging import version

from olive.constants import Precision
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.hardware.constants import ExecutionProvider
from olive.model import HfModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pass_config import BasePassConfig
from olive.search.search_parameter import Boolean, Categorical

logger = logging.getLogger(__name__)


class ModelBuilder(Pass):
    """Converts a Huggingface generative PyTorch model to ONNX model using the Generative AI builder.

    See https://github.com/microsoft/onnxruntime-genai
    """

    class BlockSize(IntEnum):
        B16 = 16
        B32 = 32
        B64 = 64
        B128 = 128
        B256 = 256

    class AccuracyLevel(IntEnum):
        fp32 = 1
        fp16 = 2
        bf16 = 3
        int8 = 4

    EP_MAP: ClassVar[dict[ExecutionProvider, str]] = {
        ExecutionProvider.CPUExecutionProvider: "cpu",
        ExecutionProvider.CUDAExecutionProvider: "cuda",
        ExecutionProvider.DmlExecutionProvider: "dml",
        ExecutionProvider.WebGpuExecutionProvider: "webgpu",
        ExecutionProvider.JsExecutionProvider: "web",
        ExecutionProvider.NvTensorRTRTXExecutionProvider: "NvTensorRtRtx",
    }

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=Precision,
                default_value=Precision.FP32,
                required=True,
                description="Precision of model.",
            ),
            "metadata_only": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description="Whether to export the model or generate required metadata only.",
            ),
            "search": PassConfigParam(
                type_=dict[str, Any], required=False, description="Search options to use for generate loop."
            ),
            "int4_accuracy_level": PassConfigParam(
                type_=ModelBuilder.AccuracyLevel,
                required=False,
                description="Specify the minimum accuracy level for activation of MatMul in int4 quantization.",
            ),
            "int4_block_size": PassConfigParam(
                type_=ModelBuilder.BlockSize,
                required=False,
                search_defaults=Categorical(
                    [ModelBuilder.BlockSize.B32, ModelBuilder.BlockSize.B64, ModelBuilder.BlockSize.B128]
                ),
                description="Specify the block_size for int4 quantization. Acceptable values: 16/32/64/128/256.",
            ),
            "int4_is_symmetric": PassConfigParam(
                type_=bool,
                required=False,
                search_defaults=Boolean(),
                description="Specify whether symmetric or asymmetric INT4 quantization needs to be used.",
            ),
            "int4_op_types_to_quantize": PassConfigParam(
                type_=list[str],
                required=False,
                description=(
                    'Specify the op types to quantize for int4 quantization. Default is None (= [ "MatMul" ]). Example:'
                    ' ["MatMul", "Gemm"]'
                ),
            ),
            "int4_nodes_to_exclude": PassConfigParam(
                type_=list[str],
                required=False,
                description="Specify when you want to exclude certain nodes from int4 quantization.",
            ),
            "int4_algo_config": PassConfigParam(
                type_=str,
                required=False,
                search_defaults=Categorical(
                    [
                        "default",
                        "rtn",
                        "k_quant_mixed",
                        "k_quant_last",
                    ]
                ),
                description="Specify the INT4 quantization algorithm to use in GenAI Model Builder",
            ),
            "use_qdq": PassConfigParam(
                type_=bool,
                required=False,
                description=(
                    "Use this option when you want to use quantize-dequantize ops. "
                    "For example, you will have a quantized MatMul op instead of the MatMulNBits op."
                ),
            ),
            "use_8bits_moe": PassConfigParam(
                type_=bool,
                required=False,
                description="Specify whether the QMoE op will use 8-bit quantization.",
            ),
            "use_webgpu_fp32": PassConfigParam(
                type_=bool,
                required=False,
                description="Specify whether to use this option to enable GPUs that do not support FP16 on WebGPU.",
            ),
            "use_cuda_bf16": PassConfigParam(
                type_=bool,
                required=False,
                description="Specify whether to use BF16 I/O for quantized models on CUDA EP.",
            ),
            "include_hidden_states": PassConfigParam(
                type_=bool,
                required=False,
                description="Specify whether to have the hidden states as an output from your ONNX model.",
            ),
            "exclude_embeds": PassConfigParam(
                type_=bool,
                required=False,
                description="Remove embedding layer from your ONNX model.",
            ),
            "exclude_lm_head": PassConfigParam(
                type_=bool,
                required=False,
                description="Remove language modeling head from your ONNX model.",
            ),
            "enable_cuda_graph": PassConfigParam(
                type_=bool,
                required=False,
                description=(
                    "The model can use CUDA graph capture for CUDA execution provider. "
                    "If enabled, all nodes being placed on the CUDA EP is the prerequisite "
                    "for the CUDA graph to be used correctly."
                ),
            ),
            "extra_options": PassConfigParam(
                type_=dict[str, Any],
                required=False,
                description="Extra key-value pairs options to pass to the model builder.",
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        # if device is GPU, but user choose CPU EP, the is_cpu should be True
        if (config.precision == Precision.FP16) and not (
            accelerator_spec.accelerator_type == Device.GPU
            and accelerator_spec.execution_provider != ExecutionProvider.CPUExecutionProvider
        ):
            logger.info("FP16 is not supported on CPU.")
            return False

        if (
            config.precision == Precision.BF16
            and accelerator_spec.execution_provider != ExecutionProvider.CUDAExecutionProvider
        ):
            logger.info("BF16 is only supported on CUDA execution provider.")
            return False

        # Support for limited precision types
        return config.precision in {Precision.FP32, Precision.FP16, Precision.BF16, Precision.INT8, Precision.INT4}

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self,
        model: Union[HfModelHandler, ONNXModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        try:
            from onnxruntime_genai.models.builder import create_model
        except ImportError:
            raise ImportError(
                "onnxruntime-genai package is required to run ModelBuilder pass. Please install the package"
                " corresponding to your onnxruntime installation using pip. cpu: onnxruntime-genai, cuda:"
                " onnxruntime-genai-cuda, directml: onnxruntime-genai-directml"
            ) from None
        self.maybe_patch_quant()

        precision = config.precision
        metadata_only = config.metadata_only

        if metadata_only:
            if not isinstance(model, ONNXModelHandler):
                raise ValueError("metadata_only option is available only with ONNXModel as input.")
        elif not isinstance(model, HfModelHandler):
            raise ValueError("model building is available only with HfModel as input.")

        Path(output_model_path).mkdir(parents=True, exist_ok=True)
        output_model_filepath = (
            Path(resolve_onnx_path(output_model_path))
            if not metadata_only
            else Path(resolve_onnx_path(output_model_path, model.onnx_file_name))
        )

        target_execution_provider = self.EP_MAP.get(self.accelerator_spec.execution_provider, "cpu")

        extra_args = {"filename": str(output_model_filepath.name)}
        if metadata_only:
            extra_args["config_only"] = True
            model_path = None
            input_path = str(model.get_resource("model_path"))
        else:
            model_path = model.model_name_or_path
            # provide the model path as input path, model builder uses input_path for quantized models
            input_path = model_path
            if model.adapter_path:
                extra_args["adapter_path"] = model.adapter_path

        extra_args.update(
            {
                key: value.value if isinstance(value, IntEnum) else value
                for key, value in config.model_dump().items()
                if value is not None and key not in {"precision", "metadata_only", "search", "extra_options"}
            }
        )

        # Override extra options with user provided in extra_options parameter
        if config.extra_options:
            extra_args.update(config.extra_options)

        # Ensure output_model_filepath matches the final filename in extra_args
        output_model_filepath = Path(output_model_path) / extra_args["filename"]

        model_attributes = copy.deepcopy(model.model_attributes or {})

        try:
            logger.debug("Building model with the following args: %s", extra_args)
            create_model(
                model_name=model_path,
                input_path=input_path,
                output_dir=str(output_model_filepath.parent),
                precision=precision,
                execution_provider=target_execution_provider,
                # model builder uses the cache_dir both as hf cache and also to store intermediate files
                # not ideal, but we can't change this without changing the model builder
                cache_dir=HF_HUB_CACHE,
                **extra_args,
            )

            # add split information if present
            split_assignments = model_attributes.get("split_assignments")
            if not metadata_only and split_assignments:
                # NOTE: currently the model builder renames modules to it's own naming convention
                # so the assignments for the renamed modules won't match
                split_assignment_str = ";".join([f"{k}={v}" for k, v in split_assignments.items()])

                # load the model and set the split_assignments as model properties
                # without the external data so that they can be used as is with the resaved model
                model_proto = onnx.load(output_model_filepath, load_external_data=False)
                onnx.helper.set_model_props(model_proto, {"split_assignments": split_assignment_str})
                onnx.save(model_proto, output_model_filepath)

            # apply layer annotations if present
            layer_annotations = model_attributes.get("layer_annotations")
            if not metadata_only and layer_annotations:
                from olive.passes.onnx.layer_annotation import annotate_proto_model

                model_proto = onnx.load(output_model_filepath, load_external_data=False)
                annotate_proto_model(model_proto, layer_annotations)
                onnx.save(model_proto, output_model_filepath)
        except Exception:
            # if model building fails, clean up the intermediate files in the cache_dir
            cache_dir = Path(HF_HUB_CACHE)
            if cache_dir.is_dir():
                for file in cache_dir.iterdir():
                    if file.suffix == ".bin":
                        file.unlink()
            raise

        # Override default search options with ones from user config
        genai_config_filepath = str(output_model_filepath.parent / "genai_config.json")
        with open(genai_config_filepath) as istrm:
            genai_config = json.load(istrm)

        genai_config["search"] = {**(genai_config.get("search") or {}), **(config.search or {})}

        with open(genai_config_filepath, "w") as ostrm:
            json.dump(genai_config, ostrm, indent=4)

        # Save HfModel config
        if isinstance(model, HfModelHandler):
            # saves the config.json and module files in the output directory
            # tokenizer and generation configs are skipped since they are already saved by the model builder
            model.save_metadata(output_model_filepath.parent)

        # add additional files generated by model builder to model_attributes
        additional_files = model_attributes.get("additional_files") or []
        if metadata_only:
            # add genai_config.json to additional_files since the model_builder creates copy of the other files
            # in the output directory leading to duplicate files in the additional_files list
            model_attributes["additional_files"] = [
                *additional_files,
                str(output_model_filepath.parent / "genai_config.json"),
            ]
        else:
            model_attributes["additional_files"] = sorted(
                set(additional_files)
                # all files in the output directory except the model and model.data files
                | {str(fp) for fp in output_model_filepath.parent.iterdir()}
                - {str(output_model_filepath), str(output_model_filepath) + ".data"}
            )

        if metadata_only:
            output_model = copy.copy(model)
            output_model.model_attributes = model_attributes
        else:
            output_model = ONNXModelHandler(
                output_model_filepath.parent,
                onnx_file_name=output_model_filepath.name,
                model_attributes=model_attributes,
            )

        return output_model

    @staticmethod
    def maybe_patch_quant():
        from onnxruntime_genai import __version__ as genai_version

        if version.parse(genai_version) < version.parse("0.9.0"):
            return

        quantized_model = importlib.import_module("onnxruntime_genai.models.quantized_model")
        quantized_model.OliveModel.__init__ = OliveQuantizedModel.__init__

        # base.py uses "from quantized_model import QuantModel" which resolves to a different module
        # because builders/ directory is in sys.path when base.py runs.
        # We need to ensure that "quantized_model" in sys.modules points to the same module we patched.
        import sys

        sys.modules["quantized_model"] = quantized_model

        builder = importlib.import_module("onnxruntime_genai.models.builder")
        builder.Model.make_packed_matmul_int4 = patched_make_packed_matmul_int4
        builder.Model.make_embedding = patched_make_embedding


class OliveQuantizedModel:
    def __init__(self, quant_type, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        logger.debug("Using OliveQuantizedModel for quantized model loading.")

        from onnxruntime_genai.models.quantized_model import QuantizedDecoderLayer, QuantizedTensorModule, TensorModule
        from safetensors.torch import load_file

        config = quant_attrs["config"]

        self.quant_type = quant_type
        self.embedding = QuantizedTensorModule() if config["embeds"] else TensorModule()
        self.final_norm = TensorModule()
        self.lm_head = (
            self.embedding
            if config["tie_word_embeddings"]
            else QuantizedTensorModule()
            if config["lm_head"]
            else TensorModule()
        )
        self.layers = [QuantizedDecoderLayer(i) for i in range(num_layers)]

        module_map = {
            "model.embed_tokens": self.embedding,
            "model.norm": self.final_norm,
            "lm_head": self.lm_head,
            **{f"model.layers.{i}": layer for i, layer in enumerate(self.layers)},
        }

        overrides = config["overrides"] or {}

        def get_layer_bits(layer_name):
            name = ".".join(layer_name.split(".")[:-1])
            return overrides.get(name, {}).get("bits", config["bits"])

        def get_layer_group_size(layer_name):
            name = ".".join(layer_name.split(".")[:-1])
            return overrides.get(name, {}).get("group_size", config["group_size"])

        def set_tensor(module, tensor_name, tensor_value, local_bits, local_group_size):
            submodule = module
            for sub_name in tensor_name.split(".")[:-1]:
                if sub_name.isdigit():
                    submodule = submodule[int(sub_name)]
                else:
                    submodule = getattr(submodule, sub_name)
            if isinstance(submodule, QuantizedTensorModule):
                for q_attr, q_value in [("bits", local_bits), ("_group_size", local_group_size)]:
                    setattr(submodule, q_attr, q_value)
                # in_features is always a multiple of group_size, group_size is a power of 2
                # assumes no padding
                if tensor_name.endswith("qweight"):
                    out_features, in_features_packed = tensor_value.shape
                    in_features = in_features_packed * 8 // local_bits
                    submodule.in_features = in_features
                    submodule.out_features = out_features
                    num_blocks = in_features // local_group_size if local_group_size != -1 else 1
                    tensor_value = tensor_value.reshape(out_features, num_blocks, -1)
            setattr(submodule, tensor_name.split(".")[-1], tensor_value)

        for weight_file in Path(input_path).iterdir():
            if weight_file.suffix == ".safetensors":
                weights = load_file(weight_file)

                # Map weights to modules
                for name, tensor in weights.items():
                    if name.endswith("inv_freq"):
                        # Skip rotary embedding weights since they can be re-calculated when looping through the model
                        continue

                    # Per-layer quantization support
                    local_bits = get_layer_bits(name)
                    local_group_size = get_layer_group_size(name)

                    prefix = ".".join(name.split(".")[:-1][:3])

                    tensor_name = (
                        name.replace(f"{prefix}.", "")
                        .replace("self_attention.", "self_attn.")
                        .replace("dense_4h_to_h.", "down_proj.")
                        .replace("dense_h_to_4h.", "gate_up_proj.")
                        .replace("query_key_value.", "qkv_proj.")
                    )
                    tensor_map = {}
                    if "qkv_proj" in tensor_name:
                        tensor_map[tensor_name.replace("qkv_proj.", "q_proj.")] = tensor[:q_size, :]
                        tensor_map[tensor_name.replace("qkv_proj.", "k_proj.")] = tensor[q_size : q_size + kv_size, :]
                        tensor_map[tensor_name.replace("qkv_proj.", "v_proj.")] = tensor[q_size + kv_size :, :]
                    elif "gate_up_proj" in tensor_name:
                        tensor_map[tensor_name.replace("gate_up_proj.", "gate_proj.")] = tensor[:intermediate_size, :]
                        tensor_map[tensor_name.replace("gate_up_proj.", "up_proj.")] = tensor[intermediate_size:, :]
                    else:
                        tensor_map[tensor_name] = tensor

                    for tensor_name, tensor_value in tensor_map.items():
                        set_tensor(module_map[prefix], tensor_name, tensor_value, local_bits, local_group_size)

        # share weights between embedding and lm head
        if isinstance(self.lm_head, TensorModule) and self.lm_head.weight is None:
            self.lm_head.weight = self.embedding.weight

        if isinstance(self.embedding, QuantizedTensorModule):
            # nest the module into .weight since the builder expects that
            class EmbeddingWrapper:
                def __init__(self, embedding):
                    self.weight = embedding

            self.embedding = EmbeddingWrapper(self.embedding)


def patched_make_packed_matmul_int4(self, q_matmul, k_matmul, v_matmul, basename, root_input, **kwargs):
    if not hasattr(q_matmul, "qweight"):
        return self.make_packed_matmul_float(q_matmul, k_matmul, v_matmul, basename, root_input, **kwargs)

    class PackedMatMul:
        def __init__(self):
            if q_matmul.bits != k_matmul.bits or q_matmul.bits != v_matmul.bits:
                raise ValueError("All MatMuls must have the same bits for packed MatMul.")
            if q_matmul.group_size != k_matmul.group_size or q_matmul.group_size != v_matmul.group_size:
                raise ValueError("All MatMuls must have the same group size for packed MatMul.")
            self.qweight = torch.cat([q_matmul.qweight, k_matmul.qweight, v_matmul.qweight], dim=0)
            self.scales = torch.cat([q_matmul.scales, k_matmul.scales, v_matmul.scales], dim=0)
            self.qzeros = (
                torch.cat([q_matmul.qzeros, k_matmul.qzeros, v_matmul.qzeros], dim=0)
                if q_matmul.qzeros is not None
                else None
            )
            self.g_idx = q_matmul.g_idx

            self.in_features = q_matmul.in_features
            self.out_features = q_matmul.out_features + k_matmul.out_features + v_matmul.out_features
            self.bits = q_matmul.bits
            self.group_size = q_matmul.group_size

    matmul = PackedMatMul()
    return self.make_matmul_int4(matmul, basename, root_input, **kwargs)


def patched_make_embedding(self, embedding):
    import onnx_ir as ir

    basename = "/model/embed_tokens"
    if getattr(self, "int4_tied_embeddings", False) or getattr(self, "shared_embeddings", False):
        logger.debug(
            "int4_tied_embedding/shared_embeddings is set to True but will be ignored. Use TieWordEmbeddings graph surgery to tie"
            " embeddings."
        )

    if hasattr(embedding, "qweight"):
        qweight = "model.embed_tokens.qweight"
        self.make_initializer(embedding.qweight.reshape([embedding.qweight.shape[0], -1]), qweight)
        scales = "model.embed_tokens.scales"
        self.make_initializer(embedding.scales, scales, to=self.io_dtype)
        if embedding.qzeros is not None:
            qzeros = "model.embed_tokens.qzeros"
            self.make_initializer(embedding.qzeros, qzeros)

        gather_name = f"{basename}/GatherBlockQuantized"
        gather_output = f"{gather_name}/output_0"
        self.make_node(
            "GatherBlockQuantized",
            inputs=[qweight, "input_ids", scales] + ([qzeros] if embedding.qzeros is not None else []),
            outputs=[gather_output],
            name=gather_name,
            domain="com.microsoft",
            bits=embedding.bits,
            block_size=embedding.group_size,
        )
    else:
        weight = "model.embed_tokens.weight"
        self.make_initializer(embedding, weight, to=self.io_dtype)

        gather_name = f"{basename}/Gather"
        gather_output = f"{gather_name}/output_0"
        self.make_node("Gather", inputs=[weight, "input_ids"], outputs=[gather_output], name=gather_name)

    self.make_value(gather_output, self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

    if self.embed_attrs["scale"] != 1:
        # Scale the embeddings
        mul_name = f"{basename}/Mul"
        mul_inputs = [gather_output, f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.embed_attrs['scale']}"]
        mul_output = f"{mul_name}/output_0"
        self.make_node("Mul", inputs=mul_inputs, outputs=[mul_output], name=mul_name)
        self.make_value(mul_output, self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

        layernorm_attrs_value = mul_output
    else:
        layernorm_attrs_value = gather_output

    if self.layernorm_attrs["cast"]["use_fp32"] and self.io_dtype != ir.DataType.FLOAT:
        # Insert output Cast node
        cast_name = f"{basename}/Cast"
        self.make_cast(
            cast_name,
            layernorm_attrs_value,
            ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        layernorm_attrs_value = f"{cast_name}/output_0"

    self.layernorm_attrs["root_input"] = layernorm_attrs_value
    self.layernorm_attrs["skip_input"] = layernorm_attrs_value
