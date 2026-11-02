#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
import torch
from safetensors import safe_open
from t8_codec import LAYOUT, encode_q2
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForCausalLM,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

QUANT_BITS = 2
QUANT_BLOCK_SIZE = 128
ZERO_POINT = 1


class _MatMulNBits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_weight, scales, zero_points, in_features, out_features):
        del ctx, packed_weight, scales, zero_points, in_features
        return x.new_zeros((*x.shape[:-1], out_features))

    @staticmethod
    def symbolic(g, x, packed_weight, scales, zero_points, in_features, out_features):
        return g.op(
            "com.microsoft::MatMulNBits",
            x,
            packed_weight,
            scales,
            zero_points,
            K_i=in_features,
            N_i=out_features,
            bits_i=QUANT_BITS,
            block_size_i=QUANT_BLOCK_SIZE,
        )


class _GatherBlockQuantized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed_weight, indices, scales, zero_points, hidden_size):
        del ctx, packed_weight, scales, zero_points
        return torch.zeros((*indices.shape, hidden_size), dtype=torch.float32, device=indices.device)

    @staticmethod
    def symbolic(g, packed_weight, indices, scales, zero_points, hidden_size):
        return g.op(
            "com.microsoft::GatherBlockQuantized",
            packed_weight,
            indices,
            scales,
            zero_points,
            gather_axis_i=0,
            quantize_axis_i=1,
            bits_i=QUANT_BITS,
            block_size_i=QUANT_BLOCK_SIZE,
        )


class _MatMulTernary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, planes, scales, in_features, out_features):
        del ctx, planes, scales, in_features
        return x.new_zeros((*x.shape[:-1], out_features))

    @staticmethod
    def symbolic(g, x, planes, scales, in_features, out_features):
        return g.op(
            "com.microsoft::MatMulTernary",
            x,
            planes,
            scales,
            K_i=in_features,
            N_i=out_features,
            format_version_i=1,
            layout_s=LAYOUT,
        )


class _LinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, unused_state, decay, beta, heads, head_dim):
        del ctx, query, key, unused_state, decay, beta, heads, head_dim
        return torch.zeros_like(value)

    @staticmethod
    def symbolic(g, query, key, value, unused_state, decay, beta, heads, head_dim):
        output_type = value.type()
        flat_shape = g.op("Constant", value_t=torch.tensor([0, 0, -1], dtype=torch.int64))
        output_shape = g.op("Constant", value_t=torch.tensor([0, 0, heads, head_dim], dtype=torch.int64))
        query = g.op("Reshape", query, flat_shape)
        key = g.op("Reshape", key, flat_shape)
        value = g.op("Reshape", value, flat_shape)
        output, _ = g.op(
            "com.microsoft::LinearAttention",
            query,
            key,
            value,
            unused_state,
            decay,
            beta,
            q_num_heads_i=heads,
            kv_num_heads_i=heads,
            update_rule_s="gated_delta",
            outputs=2,
        )
        return g.op("Reshape", output, output_shape).setType(output_type)


def _fwht(x: torch.Tensor, block_size: int) -> torch.Tensor:
    original_shape = x.shape
    x = x.reshape(-1, block_size)
    stride = 1
    while stride < block_size:
        x = x.reshape(-1, block_size // (2 * stride), 2, stride)
        left = x[:, :, 0, :]
        right = x[:, :, 1, :]
        x = torch.stack((left + right, left - right), dim=2).reshape(-1, block_size)
        stride *= 2
    return (x * (1.0 / math.sqrt(block_size))).reshape(original_shape)


def _gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    del kwargs
    if initial_state is not None or output_final_state:
        raise ValueError("The fixed-window Bonsai export does not use recurrent state inputs or outputs")
    if use_qk_l2norm_in_kernel:
        query = modeling_qwen3_5.l2norm(query, dim=-1, eps=1e-6)
        key = modeling_qwen3_5.l2norm(key, dim=-1, eps=1e-6)
    unused_state = query.new_empty((0,))
    return _LinearAttention.apply(
        query, key, value, unused_state, g, beta, int(value.shape[2]), int(value.shape[3])
    ), None


class BonsaiSwishAttention(Qwen3_5Attention):
    """The 27B checkpoint uses a swish output gate rather than Qwen3.5's sigmoid gate."""

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        del kwargs
        if past_key_values is not None:
            raise ValueError("The fixed-window Bonsai export does not use attention caches")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        query = self.q_norm(query.reshape(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = modeling_qwen3_5.apply_rotary_pos_emb(query, key, *position_embeddings)
        output, weights = modeling_qwen3_5.eager_attention_forward(
            self, query, key, value, attention_mask, scaling=self.scaling, dropout=0.0
        )
        output = output.reshape(*input_shape, -1) * torch.nn.functional.silu(gate.reshape(*input_shape, -1))
        return self.o_proj(output), weights


class PackedLinear(nn.Module):
    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        signs: torch.Tensor,
        transform_block_size: int,
    ):
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)
        self.register_buffer("signs", signs)
        self.transform_block_size = transform_block_size
        self.out_features = packed_weight.shape[0]
        self.in_features = packed_weight.shape[1] * QUANT_BLOCK_SIZE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_block_size:
            x = _fwht(x * self.signs, self.transform_block_size)
        return _MatMulNBits.apply(
            x,
            self.packed_weight,
            self.scales,
            self.zero_points,
            self.in_features,
            self.out_features,
        )


class T8Linear(nn.Module):
    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        signs: torch.Tensor,
        transform_block_size: int,
    ):
        super().__init__()
        self.out_features = packed_weight.shape[0]
        self.in_features = packed_weight.shape[1] * QUANT_BLOCK_SIZE
        planes, t8_scales = encode_q2(packed_weight.numpy(), scales.numpy(), verify=True)
        # The legacy Torch exporter cannot serialize uint32; restore the ONNX type after export.
        self.register_buffer("planes", torch.from_numpy(planes.view(np.int32)))
        self.register_buffer("scales", torch.from_numpy(t8_scales))
        self.register_buffer("signs", signs)
        self.transform_block_size = transform_block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_block_size:
            x = _fwht(x * self.signs, self.transform_block_size)
        return _MatMulTernary.apply(x, self.planes, self.scales, self.in_features, self.out_features)


class PackedEmbedding(nn.Module):
    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        signs: torch.Tensor,
        transform_block_size: int,
    ):
        super().__init__()
        packed_weight = packed_weight.reshape(packed_weight.shape[0], -1)
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)
        self.register_buffer("signs", signs)
        self.transform_block_size = transform_block_size
        self.hidden_size = packed_weight.shape[1] * (8 // QUANT_BITS)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = _GatherBlockQuantized.apply(
            self.packed_weight,
            indices,
            self.scales,
            self.zero_points,
            self.hidden_size,
        )
        if self.transform_block_size:
            x = _fwht(x, self.transform_block_size) * self.signs
        return x


class BonsaiForNextToken(nn.Module):
    def __init__(self, model: Qwen3_5ForCausalLM):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logits_index: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )[0]
        selected = torch.index_select(hidden_states, 1, logits_index)
        return self.model.lm_head(selected)


def _resolve_module(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _packed_tensors(weights, path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    words = weights.get_tensor(path + ".weight")
    scales = weights.get_tensor(path + ".scales").astype(np.float32)
    biases = weights.get_tensor(path + ".biases").astype(np.float32)
    signs = weights.get_tensor(path + ".signs").astype(np.float32)
    if not np.array_equal(biases, -scales):
        raise ValueError(f"{path}: expected ternary affine bias to equal -scale")

    out_features, words_per_row = words.shape
    in_features = words_per_row * 16
    if in_features % QUANT_BLOCK_SIZE:
        raise ValueError(f"{path}: input width {in_features} is not divisible by {QUANT_BLOCK_SIZE}")

    packed = words.view(np.uint8).reshape(
        out_features,
        in_features // QUANT_BLOCK_SIZE,
        QUANT_BLOCK_SIZE * QUANT_BITS // 8,
    )
    if any(np.any(((packed >> shift) & 3) == 3) for shift in (0, 2, 4, 6)):
        raise ValueError(f"{path}: non-ternary affine code 3")
    num_blocks = in_features // QUANT_BLOCK_SIZE
    zero_points = np.full((out_features, (num_blocks + 3) // 4), 0x55, dtype=np.uint8)
    return (
        torch.from_numpy(np.ascontiguousarray(packed)),
        torch.from_numpy(np.ascontiguousarray(scales)),
        torch.from_numpy(zero_points),
        torch.from_numpy(np.ascontiguousarray(signs)),
    )


def _assign_parameter(model: nn.Module, name: str, value: np.ndarray) -> None:
    parent, attribute = _resolve_module(model, name)
    current = getattr(parent, attribute)
    if isinstance(parent, Qwen3_5RMSNorm):
        value = value - 1.0
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    if tuple(tensor.shape) != tuple(current.shape):
        raise ValueError(f"{name}: source shape {tuple(tensor.shape)} != target shape {tuple(current.shape)}")
    parent._parameters[attribute] = nn.Parameter(tensor, requires_grad=False)


def load_bonsai_model(model_dir: Path, weight_format: str = "q2") -> Qwen3_5ForCausalLM:
    if weight_format not in ("q2", "t8"):
        raise ValueError(f"Unsupported weight format: {weight_format}")
    config_json = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config_json.get("model_type") != "prism_hadamard_qwen35":
        raise ValueError("Expected a prism_hadamard_qwen35 model")

    config = Qwen3_5TextConfig.from_dict(config_json["text_config"])
    config.use_cache = False
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
        output_gate = config_json["text_config"].get("output_gate_type", "sigmoid")
        if output_gate not in ("sigmoid", "swish"):
            raise ValueError(f"Unsupported attention output gate: {output_gate}")
        if output_gate == "swish":
            for index, layer in enumerate(model.model.layers):
                if hasattr(layer, "self_attn"):
                    layer.self_attn = BonsaiSwishAttention(config, index)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)

    packed_paths = {record["path"]: record for record in config_json["modules"]}
    embedding_path = "model.embed_tokens"
    if embedding_path not in packed_paths:
        raise ValueError("Packed embedding is missing from config.json")

    weights_path = model_dir / "model.safetensors"
    with safe_open(weights_path, framework="np") as weights:
        for path, record in packed_paths.items():
            source_path = "language_model." + path
            packed, scales, zero_points, signs = _packed_tensors(weights, source_path)
            if record["embedding"]:
                module = PackedEmbedding(packed, scales, zero_points, signs, record["block"])
            elif weight_format == "t8":
                module = T8Linear(packed, scales, signs, record["block"])
            else:
                module = PackedLinear(packed, scales, zero_points, signs, record["block"])
            parent, attribute = _resolve_module(model, path)
            setattr(parent, attribute, module)

        if config.tie_word_embeddings:
            embedding_record = packed_paths[embedding_path]
            packed, scales, zero_points, signs = _packed_tensors(weights, "language_model." + embedding_path)
            model.lm_head = (
                T8Linear(packed, scales, signs, embedding_record["block"])
                if weight_format == "t8"
                else PackedLinear(packed, scales, zero_points, signs, embedding_record["block"])
            )
        elif "lm_head" not in packed_paths:
            raise ValueError("The untied output head is missing from config.json")

        packed_prefixes = tuple("language_model." + path + "." for path in packed_paths)
        source_names = weights.keys()
        for source_name in source_names:
            if source_name.startswith(packed_prefixes):
                continue
            if not source_name.startswith("language_model."):
                continue
            target_name = source_name.removeprefix("language_model.")
            value = weights.get_tensor(source_name)
            if target_name.endswith(".conv1d.weight"):
                value = value.transpose(0, 2, 1)
            _assign_parameter(model, target_name, value)

    meta_parameters = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        raise ValueError(f"Parameters were not loaded: {meta_parameters}")
    model.eval()
    return model


def _finalize_t8_weights(model: onnx.ModelProto) -> None:
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    aliases = {node.output[0]: node.input[0] for node in model.graph.node if node.op_type == "Identity"}
    plane_names = set()
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "MatMulTernary":
            continue
        name = node.input[1]
        seen = set()
        while name in aliases:
            if name in seen:
                raise ValueError(f"T8 weight Identity cycle at {name}")
            seen.add(name)
            plane_names.add(name)
            name = aliases[name]
        plane_names.add(name)
        if name not in initializers:
            raise ValueError(f"Expected constant T8 planes for {node.name}")
        tensor = initializers[name]
        if tensor.data_type not in (onnx.TensorProto.INT32, onnx.TensorProto.UINT32):
            raise ValueError(f"Expected 32-bit T8 planes for {node.name}")
        if not tensor.raw_data and not tensor.external_data:
            raise ValueError(f"Expected raw or external T8 plane bytes for {node.name}")
        tensor.data_type = onnx.TensorProto.UINT32
    for value in model.graph.value_info:
        if value.name in plane_names:
            value.type.tensor_type.elem_type = onnx.TensorProto.UINT32


def export_model(model_dir: Path, output_path: Path, sequence_length: int, weight_format: str = "q2") -> None:
    modeling_qwen3_5.torch_chunk_gated_delta_rule = _gated_delta_rule
    model = BonsaiForNextToken(load_bonsai_model(model_dir, weight_format))
    input_ids = torch.zeros((1, sequence_length), dtype=torch.int64)
    attention_mask = torch.ones((1, sequence_length), dtype=torch.int64)
    position_ids = torch.arange(sequence_length, dtype=torch.int64).unsqueeze(0)
    logits_index = torch.tensor([sequence_length - 1], dtype=torch.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask, position_ids, logits_index),
        str(output_path),
        input_names=["input_ids", "attention_mask", "position_ids", "logits_index"],
        output_names=["logits"],
        opset_version=21,
        custom_opsets={"com.microsoft": 1},
        dynamo=False,
        external_data=True,
        do_constant_folding=False,
    )

    exported = onnx.load(output_path, load_external_data=False)
    for node in exported.graph.node:
        if node.domain == "com.microsoft" and node.op_type == "LinearAttention":
            node.input[3] = ""
    if weight_format == "t8":
        _finalize_t8_weights(exported)
        metadata = {item.key: item.value for item in exported.metadata_props}
        metadata.update(
            {
                "t8.format_version": "1",
                "t8.layout": LAYOUT,
                "t8.source": str(model_dir.resolve()),
                "t8.export": "direct_checkpoint",
                "t8.round_trip_verified": "true",
            }
        )
        onnx.helper.set_model_props(exported, metadata)
    onnx.save(exported, output_path)
    onnx.checker.check_model(output_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a PrismML Bonsai 2 MLX pack to ONNX.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing the Bonsai MLX pack.")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX model path.")
    parser.add_argument(
        "--weight-format", choices=("q2", "t8"), default="q2", help="Linear weight format; embeddings remain Q2."
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=64,
        help="Fixed sequence length for prompt and decode recomputation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    export_model(args.model_dir.resolve(), args.output.resolve(), args.sequence_length, args.weight_format)


if __name__ == "__main__":
    main()
