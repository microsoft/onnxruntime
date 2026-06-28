# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Benchmark vLLM triton_kernels.matmul_ogs for MXFP4 W4A16 on GPT-OSS shapes.

This is a runtime spike only. It uses vLLM's Hopper MXFP4 value/scale swizzles and
matmul_ogs wrapper from source via PYTHONPATH, mirroring the backend that vLLM
selects for gpt-oss on SM90 when FlashInfer/AITER are unavailable.
"""

import os
import sys

sys.path.insert(0, "/home/tianlei/vllm/vllm/third_party")

import torch
import triton
from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig, matmul_ogs
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import ExptData, RoutingData
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def swizzle_mxfp4_for_hopper(quant_tensor, scale, num_warps=8):
    value_layout, value_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout, scale_opts = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=num_warps)
    # vLLM expects incoming quant_tensor [E, N, K/2] and scale [E, N, K/32], then transposes.
    w = convert_layout(wrap_torch_tensor(quant_tensor.transpose(-2, -1), dtype=FP4), value_layout, **value_opts)
    s = convert_layout(wrap_torch_tensor(scale.transpose(-2, -1)), scale_layout, **scale_opts)
    return w, PrecisionConfig(weight_scale=s, flex_ctx=FlexCtx(rhs_data=InFlexData()))


def make_mxfp4(E, N, K, dev):
    codes = torch.randint(0, 16, (E, N, K), device=dev, dtype=torch.int32)
    scale_codes = torch.randint(120, 128, (E, N, K // 32), device=dev, dtype=torch.uint8)
    packed = (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).to(torch.uint8)
    lut = _E2M1.to(dev)
    scales = torch.where(scale_codes == 0, 0.0, torch.pow(torch.tensor(2.0, device=dev), scale_codes.float() - 127.0))
    dense = (lut[codes] * scales.repeat_interleave(32, dim=2)).to(torch.bfloat16)
    return packed, scale_codes, dense


def bench(M, N, K, E=1):
    dev = "cuda"
    x = (torch.randn(M, K, device=dev) * 0.05).to(torch.bfloat16)
    packed, scale_codes, dense_w = make_mxfp4(E, N, K, dev)
    w, pc = swizzle_mxfp4_for_hopper(packed, scale_codes)
    y = torch.empty((1, M, N), device=dev, dtype=torch.bfloat16)
    out = matmul_ogs(x, w, None, precision_config=pc, y=y)
    ref = (x.float() @ dense_w[0].T.float()).to(torch.bfloat16)
    err = (out.float() - ref.float()).abs().max().item()
    rel = err / (ref.float().abs().max().item() + 1e-6)
    ms = triton.testing.do_bench(lambda: matmul_ogs(x, w, None, precision_config=pc, y=y))
    print(f"vLLM matmul_ogs M={M} N={N} K={K} E={E} | rel={rel:.4f} | {ms*1000:.1f} us")


def make_decode_routing(M, E, dev):
    hist = torch.zeros(E, device=dev, dtype=torch.int32)
    hist[:M] = 1
    token_offs_raw = torch.empty(E, device=dev, dtype=torch.int32)
    token_offs_raw[:M] = torch.arange(M, device=dev, dtype=torch.int32)
    token_offs_raw[M:] = M
    block_pid_map = torch.arange(M, device=dev, dtype=torch.int32)
    expt_data = ExptData(
        hist=hist,
        token_offs_raw=token_offs_raw,
        token_offs_pad={16: token_offs_raw, 32: token_offs_raw, 64: token_offs_raw, 128: token_offs_raw},
        block_pid_map={16: block_pid_map, 32: block_pid_map, 64: block_pid_map, 128: block_pid_map},
    )
    return RoutingData(None, hist, E, 1, expt_data, expected_tokens_per_expt=1)


def bench_decode_routed(M, N, K, E=32):
    dev = "cuda"
    x = (torch.randn(M, K, device=dev) * 0.05).to(torch.bfloat16)
    packed, scale_codes, dense_w = make_mxfp4(E, N, K, dev)
    w, pc = swizzle_mxfp4_for_hopper(packed, scale_codes)
    routing = make_decode_routing(M, E, dev)
    y = torch.empty((1, M, N), device=dev, dtype=torch.bfloat16)
    out = matmul_ogs(x, w, None, routing_data=routing, precision_config=pc, y=y).view(M, N)
    ref = torch.stack([x[i].float() @ dense_w[i].T.float() for i in range(M)]).to(torch.bfloat16)
    err = (out.float() - ref.float()).abs().max().item()
    rel = err / (ref.float().abs().max().item() + 1e-6)
    ms = triton.testing.do_bench(lambda: matmul_ogs(x, w, None, routing_data=routing, precision_config=pc, y=y))
    print(f"vLLM routed M={M} N={N} K={K} E={E} | rel={rel:.4f} | {ms*1000:.1f} us")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    bench(M=4, N=5760, K=2880)
    bench(M=4, N=2880, K=2880)
    bench(M=16, N=5760, K=2880)
    bench_decode_routed(M=4, N=5760, K=2880)
    bench_decode_routed(M=4, N=2880, K=2880)
