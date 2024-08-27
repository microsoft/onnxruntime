# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Any, List, Tuple

import os
import gc
import itertools
import random
import unittest

import numpy as np
import torch
import torch.distributions as dist
import pytest
from parameterized import parameterized

from onnx import TensorProto, helper
from onnxruntime import InferenceSession, OrtValue, SessionOptions


DTYPES = ["float16", "float32", "float8_e4m3fn"]
NUM_GEN_SEQS = [1, 7]
NUM_HEADS = [
  (19, 19),  # mha
  (16, 4),   # gqa4
  (32, 2),   # gqa8 with 2x broadcasting
]
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
PAGE_SIZES = [8, 16, 32]


def generate_page_mapping(
    max_num_pages: int,
    num_seqs: int,
    max_context_len: int,
    page_size: int,
):
    unique_page_mapping = [i for i in range(max_num_pages)]
    max_num_pages_per_seq = (max_context_len + page_size - 1) // page_size
    random.shuffle(unique_page_mapping)
    page_table = []
    for i in range(num_seqs):
        page_table.append(unique_page_mapping[i * max_num_pages_per_seq : (i + 1) * max_num_pages_per_seq])
    assert len(page_table[-1]) == max_num_pages_per_seq, "alloc more pages to allow generating unique page mapping"
    return page_table


def get_useful_slots(
    fill_non_used: float | None,
    page_size: int,
    page_table: list[list[int]],
    context_lens: list[int],
):
    useful_slots = None
    if fill_non_used is not None:
        useful_slots = []
        for seq, end in enumerate(context_lens):
            seq_num_pages = (end - 1) // page_size + 1
            seq_useful_slots = []
            for logical_pid in range(seq_num_pages):
                physical_pid = page_table[seq][logical_pid]
                seq_useful_slots.extend(list(range(physical_pid * page_size, physical_pid * page_size + page_size)))
            useful_slots.extend(seq_useful_slots[:end])
    return useful_slots


def get_dtypes(kv_dtype) -> Tuple[Any]:
    if isinstance(kv_dtype, str):
        kv_dtype = getattr(torch, kv_dtype)
    mapping = {
        None: (torch.float16, torch.float16, None),
        torch.float16: (torch.float16, torch.float16, None),
        torch.float32: (torch.float32, torch.float32, None),
        torch.float8_e4m3fn: (torch.float16, torch.float8_e4m3fn, torch.float16),
    }
    return mapping.get(kv_dtype)


def masked_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", q, k).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(v.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, v)
    return out


def reconstruct_kv(
    page_mapping_of_seq: torch.Tensor,
    context_len: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_scalebias: torch.Tensor | None = None,
    chunksize=32,
):
    #  (num_pages, num_heads, head_size//chunksize, page_size)
    k_padded = k_cache.view(torch.int8).index_select(0, page_mapping_of_seq).view(k_cache.dtype).clone()
    v_padded = v_cache.view(torch.int8).index_select(0, page_mapping_of_seq).view(v_cache.dtype).clone()
    #  (num_pages, 2, num_heads, 2, head_size//chunksize, page_size)
    if kv_scalebias is not None:
        sb_padded = kv_scalebias.index_select(0, page_mapping_of_seq).clone()

    k_padded = k_padded.moveaxis(0, -3)  #  (num_heads, head_size//x, num_pages, page_size, x)
    k_padded = k_padded.moveaxis(-1, -3)  # (num_heads, head_size//x, x, num_pages, page_size)
    v_padded = v_padded.moveaxis(0, -2)  #  (num_heads, head_size,    num_pages, page_size)
    # print(k_padded.shape)
    # print(v_padded.shape)
    if kv_scalebias is not None:
        sb_padded = sb_padded.moveaxis(0, -2)  #  (2, num_heads, 2, head_size//chunksize, num_pages, page_size)
        # print(sb_padded.shape)

    k_padded = k_padded.reshape(
        (k_padded.shape[0], k_padded.shape[1] * k_padded.shape[2], k_padded.shape[3] * k_padded.shape[4])
    )
    v_padded = v_padded.reshape(v_padded.shape[:2] + (v_padded.shape[2] * v_padded.shape[3],))
    # print(k_padded.shape)
    # print(v_padded.shape)
    if kv_scalebias is not None:
        sb_padded = sb_padded.reshape(sb_padded.shape[:4] + (sb_padded.shape[4] * sb_padded.shape[5],))
        sb_padded = sb_padded.repeat_interleave(chunksize, dim=3)
        # print(sb_padded.shape)

    # (num_heads, head_size, context_len)
    k = k_padded[:, :, :context_len]
    v = v_padded[:, :, :context_len]
    # print(k.shape, k.dtype)
    # print(v.shape, v.dtype)
    if kv_scalebias is not None:
        valid_head_size = k.shape[1]
        sb = sb_padded[:, :, :, :valid_head_size, :context_len]
        k = sb[0, :, 0] * k.to(sb.dtype) + sb[0, :, 1]
        v = sb[1, :, 0] * v.to(sb.dtype) + sb[1, :, 1]
        # print("k scale", sb[0, :, 0].max())
        # print("k bias ", sb[0, :, 1].max())
        # print("v scale", sb[1, :, 0].max())
        # print("v bias ", sb[1, :, 1].max())

    # (context_len, num_heads, head_size)
    original_k = k.moveaxis(-1, 0).contiguous()
    original_v = v.moveaxis(-1, 0).contiguous()
    if kv_scalebias is None:
        # print(original_k.shape, original_v.shape)
        return original_k, original_v
    else:
        sb_chunked = sb[:, :, :, ::chunksize].moveaxis(-1, 0).contiguous()
        # print(original_k.shape, original_v.shape, sb_chunked.shape)
        return original_k, original_v, sb_chunked


def reconstruct_kv_then_mha(
    output: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_scalebias: torch.Tensor | None,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: torch.Tensor | None,
) -> None:
    num_q_heads = q.shape[1]
    num_seqs = q.shape[0]
    _, num_kv_heads, head_size, page_size = v_cache.shape
    context_lens = context_lens.cpu().tolist()
    if kv_scalebias is not None:
        num_chunks = kv_scalebias.shape[-2]
        chunk_size = head_size // num_chunks

        assert k_cache.dtype == torch.float8_e4m3fn
        assert v_cache.dtype == torch.float8_e4m3fn

    for i in range(num_seqs):
        context_len = int(context_lens[i])
        num_pages = (context_len - 1) // page_size + 1
        page_mapping_of_seq = page_table[i].to(torch.int64)[:num_pages]
        reconstructed_kv = reconstruct_kv(page_mapping_of_seq, context_len, k_cache, v_cache, kv_scalebias)
        k = reconstructed_kv[0]
        v = reconstructed_kv[1]

        num_query_heads_per_kv_head = num_q_heads // num_kv_heads
        assert num_q_heads % num_kv_heads == 0
        if num_query_heads_per_kv_head > 1:
            k = torch.repeat_interleave(k, num_query_heads_per_kv_head, dim=1)
            v = torch.repeat_interleave(v, num_query_heads_per_kv_head, dim=1)

        attn_bias = None
        if alibi_slopes is not None:
            position_ids = torch.arange(context_len, device="cuda").int()
            attn_bias = (position_ids - context_len + 1).float()
            attn_bias = alibi_slopes.view(-1, 1, 1) * attn_bias.view(1, 1, -1)

        out = masked_mha(q[i].unsqueeze(0), k, v, scale, attn_bias)
        out = out.view(num_q_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    torch.cuda.synchronize()


def gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=False):
    if multidist:
        num_dist = num_pages * num_heads * page_size

        b = (torch.rand(num_dist).cuda().abs() * 16 + 0.0005) * scale
        a = torch.zeros_like(b)
        a = a.uniform_(-4, 4) * scale

        uniform_dist = dist.Normal(a, b)  # a batch of Uniform distributions
        samples = uniform_dist.sample((head_size,))  # (head_size, num_dist)

        kv_data = samples.reshape(
            (head_size, num_heads, num_pages, page_size)
        )  # (head_size, num_heads, num_pages, page_size)
        kv_data = kv_data.swapaxes(0, 2)  # (num_pages, num_heads, head_size, page_size)
    else:
        kv_data = torch.empty(size=(num_pages, num_heads, head_size, page_size), dtype=torch.float32, device="cuda")
        kv_data = kv_data.uniform_(-scale, scale)
    return kv_data.contiguous()


def div_x(tensor, x):
    shape = tensor.shape  # (num_pages, num_heads, head_size, page_size)
    assert shape[-2] % x == 0
    tensor = tensor.reshape(
        shape[:-2] + (shape[-2] // x, x) + shape[-1:]
    )  # (num_pages, num_heads, head_size//x, x, page_size)
    return torch.swapaxes(tensor, -1, -2).contiguous()  # (num_pages, num_heads, head_size//x, page_size, x)


def fill_unused_slots(kv_data, useful_slots, fill_value):
    # kv_data.shape (num_pages, num_heads, head_size, page_size)
    kv_data = kv_data.swapaxes(0, 2)  # (head_size, num_heads, num_pages, page_size)
    old_shape = kv_data.shape
    kv_data = kv_data.reshape(kv_data.shape[:2] + (-1,))
    kv_data_new = torch.full_like(kv_data, fill_value=fill_value)
    kv_data_new[:, :, useful_slots] = kv_data[:, :, useful_slots]
    kv_data_new = kv_data_new.reshape(old_shape)
    return kv_data_new.swapaxes(0, 2).contiguous()


def create_kv_caches(
    num_pages: int,
    page_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    kv_dtype: torch.dtype,
    useful_slots=None,
    fill_value=None,
    multidist_key=True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    gc.collect()

    dtype = kv_dtype
    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()

    key_caches = []
    for _ in range(num_layers):
        key_cache = gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=multidist_key).to(kv_dtype)
        if useful_slots is not None:
            key_cache = fill_unused_slots(key_cache, useful_slots, fill_value)
        key_cache = div_x(key_cache, x)
        key_caches.append(key_cache)

    value_caches = []
    for _ in range(num_layers):
        value_cache = gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=False).to(kv_dtype)
        if useful_slots is not None:
            value_cache = fill_unused_slots(value_cache, useful_slots, fill_value)
        value_caches.append(value_cache)

    return key_caches, value_caches


def pad_to_chunksize(tensor, chunksize) -> Tuple[bool, torch.Tensor]:
    def ceil_div(x, y):
        return (x - 1) // y + 1

    shape = tensor.shape  # (num_pages, num_heads, head_size, page_size)
    needs_pad = shape[2] % chunksize != 0
    if needs_pad:
        padder_shape = [shape[0], shape[1], ceil_div(shape[2], chunksize) * chunksize - shape[2], shape[3]]
        tensor_padder = torch.ones(padder_shape, dtype=tensor.dtype, device=tensor.device) * float("nan")
        tensor = torch.cat([tensor, tensor_padder], dim=2)
    return needs_pad, tensor


def get_scaled_and_scalebias(tensor, chunksize, scaled_dtype, scalebias_dtype):
    dummy = False
    # dummy = True  # FIXME: dummy values
    target_max = torch.finfo(scaled_dtype).max - 32
    if dummy:
        target_max = 400
    print("[get_scaled_and_scalebias] target_max =", target_max)

    original_head_size = tensor.shape[2]
    is_padded, tensor = pad_to_chunksize(tensor, chunksize)
    shape = tensor.shape  # (num_pages, num_heads, head_size, page_size)
    # (num_pages, num_heads, ceil_div(head_size,chunksize), chunksize, page_size)
    chunked_shape = [shape[0], shape[1], shape[2] // chunksize, chunksize, shape[3]]
    tensor = tensor.clone().to(torch.float32)
    tensor = tensor.reshape(chunked_shape)
    masked_inf = tensor.isinf()
    tensor[masked_inf] = float("nan")
    mean = torch.nanmean(tensor, -2, True)
    if dummy:
        mean = torch.zeros_like(mean)

    shifted = tensor - mean
    if is_padded:
        shifted = torch.nan_to_num(shifted, 0)

    amax, _ = torch.max(shifted.abs(), -2, True)
    if dummy:
        amax = 0.1 * torch.ones_like(amax)

    scaled = (shifted / amax) * target_max
    scaled[masked_inf] = float("inf")
    scaled = scaled.reshape(shape)
    if is_padded:
        scaled = scaled[:, :, :original_head_size, :]
    scaled = scaled.to(scaled_dtype).contiguous()

    # original = a * scaled + b
    b = mean
    a = amax / target_max

    # (num_pages, num_heads, ceil_div(head_size, chunksize), 1, page_size) -> (num_pages, num_heads, ceil_div(head_size, chunksize), page_size)
    a = torch.squeeze(a, -2)
    b = torch.squeeze(b, -2)

    scalebias = torch.stack([a, b], dim=2).unsqueeze(1).to(scalebias_dtype)

    return scaled, scalebias


def create_fp8_kv_caches(
    num_pages: int,
    page_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    kv_dtype: torch.dtype,
    kv_scalebias_dtype=torch.float16,
    kv_scalebias_chunk_size: int = 32,
    useful_slots=None,
    fill_value=None,
    multidist_key=True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    gc.collect()
    assert kv_dtype in (torch.float8_e4m3fn,)
    # assert head_size % kv_scalebias_chunk_size == 0

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=kv_dtype).element_size()

    key_caches = []
    value_caches = []
    kv_scalebias = []
    for _ in range(num_layers):
        k_cache = gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=multidist_key)
        v_cache = gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=False)

        if useful_slots is not None:
            k_cache = fill_unused_slots(k_cache, useful_slots, fill_value)
            v_cache = fill_unused_slots(v_cache, useful_slots, fill_value)

        k_cache, k_scalebias = get_scaled_and_scalebias(k_cache, kv_scalebias_chunk_size, kv_dtype, kv_scalebias_dtype)
        v_cache, v_scalebias = get_scaled_and_scalebias(v_cache, kv_scalebias_chunk_size, kv_dtype, kv_scalebias_dtype)

        k_cache = div_x(k_cache, x)

        key_caches.append(k_cache)
        value_caches.append(v_cache)

        scalebias = torch.cat([k_scalebias, v_scalebias], dim=1)
        kv_scalebias.append(scalebias)

    return key_caches, value_caches, kv_scalebias


def create_kv_caches_wrapper(*args, **kwargs):
    if kwargs.get("kv_dtype", None) == torch.float8_e4m3fn:
        return create_fp8_kv_caches(*args, **kwargs)
    else:
        kwargs.pop("kv_scalebias_dtype", None)
        key_caches, value_caches = create_kv_caches(*args, **kwargs)
        dummy_scalebias = [None] * len(key_caches[0])
        return key_caches, value_caches, dummy_scalebias


def torch_dtype_to_tensor_proto(dtype):
    return {
        torch.bfloat16: TensorProto.BFLOAT16,
        torch.float32: TensorProto.FLOAT,
        torch.float16: TensorProto.FLOAT16,
        torch.float8_e4m3fn: TensorProto.FLOAT8E4M3FN,
        torch.int32: TensorProto.INT32,
    }[dtype]


def torch_dtype_to_numpy_dtype(dtype):
    return {
        torch.bfloat16: None,
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.float8_e4m3fn: None,
        torch.int32: np.int32,
    }[dtype]


class TestMHA(unittest.TestCase):

    def run_paged_attention_opkernel_with_torch_tensor(
        self,
        output,
        q,
        k_cache,
        v_cache,
        kv_scalebias,
        num_kv_heads,
        scale,
        page_table,
        context_lens,
        page_size,
        max_context_len,
        alibi_slopes,
    ):
        num_seqs, num_heads, head_size = q.size()
        assert output.size() == q.size()
        num_pages, num_kv_heads, head_size, _ = v_cache.size()
        assert v_cache.size(3) == page_size
        assert k_cache.size(0) == num_pages
        assert k_cache[0].numel() == page_size * num_kv_heads * head_size
        group_size = 32  # NOTE: only support group_size 32 for now
        if kv_scalebias is not None:
            assert kv_scalebias.size(0) == num_pages
            assert kv_scalebias[0].numel() == page_size * num_kv_heads * head_size // 32
        assert page_table.size(0) == num_seqs
        max_num_pages_per_sequence = page_table.size(1)
        assert context_lens.size() == (num_seqs,)
        if alibi_slopes is not None:
            assert alibi_slopes.size() == (num_heads,)

        nodes = [
            helper.make_node(
                "PagedAttention",
                [
                    "query",
                    "key_cache",
                    "value_cache",
                    "page_table",
                    "context_lens",
                    "max_context_len",
                    "" if alibi_slopes is None else "alibi_bias",
                    "" if kv_scalebias is None else "kv_quant_param",
                ],
                ["output"],
                "PagedAttention_0",
                scale=scale,
                page_size=page_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                domain="com.microsoft",
            ),
        ]
        graph_input = [
            helper.make_tensor_value_info(
                "query",
                torch_dtype_to_tensor_proto(q.dtype),
                [num_seqs, num_heads, head_size],
            ),
            helper.make_tensor_value_info(
                "key_cache",
                torch_dtype_to_tensor_proto(k_cache.dtype),
                [num_pages, page_size * num_kv_heads * head_size],
            ),
            helper.make_tensor_value_info(
                "value_cache",
                torch_dtype_to_tensor_proto(v_cache.dtype),
                [num_pages, page_size * num_kv_heads * head_size],
            ),
            helper.make_tensor_value_info(
                "page_table",
                torch_dtype_to_tensor_proto(page_table.dtype),
                [num_seqs, max_num_pages_per_sequence],
            ),
            helper.make_tensor_value_info(
                "context_lens",
                torch_dtype_to_tensor_proto(context_lens.dtype),
                [num_seqs],
            ),
            helper.make_tensor_value_info(
                "max_context_len",
                TensorProto.INT32,
                [1],
            ),
            helper.make_tensor_value_info(
                "alibi_bias",
                TensorProto.FLOAT,
                [num_heads],
            ),
            helper.make_tensor_value_info(
                "kv_quant_param",
                TensorProto.FLOAT16,
                [num_pages, 2 * page_size * num_kv_heads * head_size // group_size],
            ),
        ]

        graph_output = [
            helper.make_tensor_value_info(
                "output",
                torch_dtype_to_tensor_proto(output.dtype),
                [num_seqs, num_heads, head_size],
            ),
        ]

        graph = helper.make_graph(
            nodes,
            "PagedAttention_Graph",
            graph_input,
            graph_output,
        )

        model = helper.make_model(graph)
        sess_options = SessionOptions()

        ort_session = InferenceSession(model.SerializeToString(), sess_options, providers=["CUDAExecutionProvider"])
        # We do not have a way to test fp8 kv cache as we cannot pass FLOAT8E4M3FN tensors as inputs into the session
        testable = torch_dtype_to_tensor_proto(k_cache.dtype) != TensorProto.FLOAT8E4M3FN
        if not testable:
            pytest.skip("unable to test")

        k_cache_packed = k_cache.reshape((num_pages, -1))
        v_cache_packed = v_cache.reshape((num_pages, -1))
        max_context_len_tensor = np.array([max_context_len], dtype=np.int32)

        use_io_binding = True
        if use_io_binding:
            io_binding = ort_session.io_binding()

            def bind_torch(io_binding, name, torch_tensor):
                if torch_tensor is None:
                    return
                io_binding.bind_input(
                    name,
                    "cuda",
                    0,
                    torch_dtype_to_numpy_dtype(torch_tensor.dtype),
                    tuple(torch_tensor.shape),
                    torch_tensor.data_ptr(),
                )

            bind_torch(io_binding, "query", q)
            bind_torch(io_binding, "key_cache", k_cache_packed)
            bind_torch(io_binding, "value_cache", v_cache_packed)
            bind_torch(io_binding, "page_table", page_table)
            bind_torch(io_binding, "context_lens", context_lens)
            io_binding.bind_cpu_input("max_context_len", max_context_len_tensor)
            if alibi_slopes is not None:
                bind_torch(io_binding, "alibi_bias", alibi_slopes)
            if kv_scalebias is not None:
                bind_torch(io_binding, "kv_quant_param", kv_scalebias)

            io_binding.bind_output("output")

            ort_session.run_with_iobinding(io_binding)
            outputs = io_binding.copy_outputs_to_cpu()
            output[:] = torch.tensor(outputs[0]).cuda()

        else:
            outputs = ort_session.run(
                ["output"],
                {
                    "query": q.cpu().numpy(),
                    "key_cache": k_cache_packed.reshape((num_pages, -1)).cpu().numpy(),
                    "value_cache": v_cache_packed.cpu().numpy(),
                    "page_table": page_table.cpu().numpy(),
                    "context_lens": context_lens.cpu().numpy(),
                    "max_context_len": max_context_len_tensor,
                    "alibi_bias": None if alibi_slopes is None else alibi_slopes.cpu().numpy(),
                    "kv_quant_param": (
                        None if kv_scalebias is None else kv_scalebias.reshape((num_pages, -1)).cpu().numpy()
                    ),
                },
            )
            output[:] = torch.tensor(outputs[0]).cuda()

    @torch.inference_mode()
    def _test_paged_attention(
        self,
        num_seqs: int,
        seq_len: int | None,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
        page_size: int,
        kv_dtype: str | torch.dtype | None,
        use_alibi: bool,
        seed: int = int(os.environ.get("SEED", "0")),
        max_seq_len: int = 4096,
        max_num_pages: int = 5000,
        fill_non_used: float | None = None,
    ):
        dtype, kv_dtype, kv_scalebias_dtype = get_dtypes(kv_dtype)
        print(dtype, kv_dtype, kv_scalebias_dtype)

        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        scale = float(1.0 / (head_size**0.5))
        q = torch.empty(num_seqs, num_q_heads, head_size, dtype=dtype, device="cuda")
        q.uniform_(-scale, scale)
        # q = torch.randn((num_seqs, num_q_heads, head_size), dtype=dtype, device="cuda")

        assert num_q_heads % num_kv_heads == 0

        alibi_slopes = None
        if use_alibi:
            alibi_slopes = torch.randn(num_q_heads, dtype=torch.float, device="cuda") * scale * 4

        if seq_len is None:
            seq_len = os.getenv("SEQ_LEN", None)
            seq_len = int(seq_len) if seq_len is not None else None
        if seq_len is None:
            context_lens = [random.randint(1, max_seq_len) for _ in range(num_seqs)]
        elif isinstance(seq_len, int):
            context_lens = [seq_len for _ in range(num_seqs)]
        else:
            context_lens = seq_len
            assert len(context_lens) == num_seqs
        max_context_len = max(context_lens)
        context_lens_py = context_lens
        context_lens = torch.tensor(context_lens_py, dtype=torch.int, device="cuda")
        print("context_lens:", context_lens_py)

        page_table_py = generate_page_mapping(max_num_pages, num_seqs, max_context_len, page_size)
        page_table = torch.tensor(page_table_py, dtype=torch.int, device="cuda")
        print("page_table:", page_table_py)

        useful_slots = get_useful_slots(fill_non_used, page_size, page_table_py, context_lens_py)

        k_caches, v_caches, kv_scalebiases = create_kv_caches_wrapper(
            max_num_pages,
            page_size,
            1,
            num_kv_heads,
            head_size,
            kv_dtype=kv_dtype,
            kv_scalebias_dtype=kv_scalebias_dtype,
            useful_slots=useful_slots,
            fill_value=fill_non_used,
        )
        k_cache, v_cache, kv_scalebias = k_caches[0], v_caches[0], kv_scalebiases[0]

        if kv_dtype == torch.float8_e4m3fn:
            assert kv_scalebias is not None
        else:
            assert kv_scalebias_dtype is None
            assert kv_scalebias is None

        output = torch.empty_like(q)
        self.run_paged_attention_opkernel_with_torch_tensor(
            output,
            q,
            k_cache,
            v_cache,
            kv_scalebias,
            num_kv_heads,
            scale,
            page_table,
            context_lens,
            page_size,
            max_context_len,
            alibi_slopes,
        )

        ref_output = torch.empty_like(q)
        reconstruct_kv_then_mha(
            ref_output,
            q,
            k_cache,
            v_cache,
            kv_scalebias,
            page_table,
            context_lens,
            scale,
            alibi_slopes,
        )
        torch.cuda.synchronize()

        if bool(int(os.environ.get("DUMP_RESULTS", "0"))):
            print(ref_output)
            diff = output - ref_output
            print(diff.abs().max())
            print(diff)

            import numpy as np

            np.save("ref.npy", ref_output.cpu().numpy())
            np.save("our.npy", output.cpu().numpy())

        if dtype == torch.float32:
            assert torch.allclose(output, ref_output, atol=1e-5, rtol=1e-6)
        else:
            assert torch.allclose(output, ref_output, atol=2e-4, rtol=2e-5)

    @parameterized.expand(
        itertools.product(
            NUM_GEN_SEQS,
            [None],
            NUM_HEADS,
            HEAD_SIZES,
            PAGE_SIZES,
            [dtype for dtype in DTYPES if "float8" not in dtype],
            [False, True],
        )
    )
    def test_paged_attention(
        self,
        num_seqs: int,
        seq_len: int | None,
        num_heads: Tuple[int, int],
        head_size: int,
        page_size: int,
        kv_dtype: str,
        use_alibi: bool,
    ):
        num_q_heads, num_kv_heads = num_heads
        self._test_paged_attention(
            num_seqs,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_size,
            page_size,
            kv_dtype,
            use_alibi,
        )


if __name__ == "__main__":
    unittest.main()
