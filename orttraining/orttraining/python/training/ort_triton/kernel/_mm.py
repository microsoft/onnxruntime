# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
import os
from types import ModuleType
from typing import Tuple

import torch

from .._cache import ModuleCache, PyCodeCache
from .._utils import next_power_of_2

_DEBUG_MODE = "ORTMODULE_TRITON_DEBUG" in os.environ and int(os.getenv("ORTMODULE_TRITON_DEBUG")) == 1


_MM_TEMPLATE = """
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
{autotune_configs}
    ],
    key=["M", "N", "K"],
)
@triton.jit
def {kernel_name}(
    A, B, OUT, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    M = {M}
    N = {N}
    K = {K}
    stride_am = {stride_am}
    stride_ak = {stride_ak}
    stride_bk = {stride_bk}
    stride_bn = {stride_bn}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        if {even_k}:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32={allow_tf32})
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    OUT = OUT + (idx_m * N + idx_n)
    tl.store(OUT, {post_process}, mask=mask)


def {func_name}(a, b, out):
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv({M}, META["BLOCK_M"]) * triton.cdiv({N}, META["BLOCK_N"]),)
    {kernel_name}[grid](a, b, out, {M}, {N}, {K})
"""


_BMM_TEMPLATE = """
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
{autotune_configs}
    ],
    key=["M", "N", "K"],
)
@triton.jit
def {kernel_name}(
    A, B, OUT, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    M = {M}
    N = {N}
    K = {K}
    stride_aq = {stride_aq}
    stride_am = {stride_am}
    stride_ak = {stride_ak}
    stride_bq = {stride_bq}
    stride_bk = {stride_bk}
    stride_bn = {stride_bn}
    stride_cq = M * N

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        if {even_k}:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32={allow_tf32})
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    OUT = OUT + (idx_m * N + idx_n + idx_q * stride_cq)
    tl.store(OUT, {post_process}, mask=mask)


def {func_name}(a, b, out):
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv({M}, META["BLOCK_M"]) * triton.cdiv({N}, META["BLOCK_N"]), {batch}, 1)
    {kernel_name}[grid](a, b, out, {M}, {N}, {K})
"""


_GEMM_TEMPLATE = """
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
{autotune_configs}
    ],
    key=["M", "N", "K"],
)
@triton.jit
def {kernel_name}(
    A, B, C, OUT, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr
):
    M = {M}
    N = {N}
    K = {K}
    stride_am = {stride_am}
    stride_ak = {stride_ak}
    stride_bk = {stride_bk}
    stride_bn = {stride_bn}
    stride_cm = {stride_cm}
    stride_cn = {stride_cn}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        if {even_k}:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32={allow_tf32})
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    C = C + (idx_m * stride_cm + idx_n * stride_cn)
    c = tl.load(C, mask=mask, other=0.)
    OUT = OUT + (idx_m * N + idx_n)
    tl.store(OUT, {post_process} + c * {beta}, mask=mask)


def {func_name}(a, b, c, out):
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv({M}, META["BLOCK_M"]) * triton.cdiv({N}, META["BLOCK_N"]),)
    {kernel_name}[grid](a, b, c, out, {M}, {N}, {K})
"""


def _mm_configs(dtype, m, n, k, trans_a, trans_b, alpha, func_name):
    condidates = [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        (32, 32, 16, 1, 2),
        (64, 64, 16, 2, 4),
        (64, 64, 32, 2, 4),
        (64, 128, 32, 3, 4),
        (128, 64, 32, 3, 4),
        (64, 128, 32, 4, 8),
        (128, 64, 32, 4, 8),
        (64, 32, 32, 5, 8),
        (32, 64, 32, 5, 8),
        (128, 128, 32, 2, 8),
        (64, 64, 64, 3, 8),
        (32, 32, 128, 2, 4),
    ]
    tm = max(next_power_of_2(m), 16)
    tn = max(next_power_of_2(n), 16)
    tk = max(next_power_of_2(k), 16)
    config_set = set()
    config_strs = []
    max_bk = 1
    for bm, bn, bk, num_stages, num_warps in condidates:
        new_bm = min(bm, tm)
        new_bn = min(bn, tn)
        new_bk = min(bk, tk)
        if (new_bm, new_bn, new_bk) in config_set:
            continue
        config_set.add((new_bm, new_bn, new_bk))
        if new_bk > max_bk:
            max_bk = new_bk
        new_num_warps = min(num_warps, new_bm * new_bn // 256)
        config_strs.append(
            f'        triton.Config({{"BLOCK_M": {new_bm}, "BLOCK_N": {new_bn}, "BLOCK_K": {new_bk}, "GROUP_M": 8}}, '
            f"num_stages={num_stages}, num_warps={new_num_warps}),"
        )
    autotune_configs = "\n".join(config_strs)
    post_process = "acc" if alpha == 1.0 else f"acc * {alpha}"
    if dtype == torch.float16:
        if alpha != 1.0:
            post_process = f"({post_process})"
        post_process = f"{post_process}.to(tl.float16)"
    return dict(
        autotune_configs=autotune_configs,
        kernel_name=f"kernel_{func_name}",
        M=m,
        N=n,
        K=k,
        stride_am=(1 if trans_a else k),
        stride_ak=(m if trans_a else 1),
        stride_bk=(1 if trans_b else n),
        stride_bn=(k if trans_b else 1),
        even_k=(k % max_bk == 0),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        post_process=post_process,
        func_name=func_name,
    )


def _gen_mm_key(dtype: torch.dtype, m: int, n: int, k: int, trans_a: bool, trans_b: bool, alpha: float) -> int:
    return hash(f"mm|{dtype}|{m}|{n}|{k}|{trans_a}|{trans_b}|{alpha}") % (10**8)


def _gen_mm_module(
    dtype: torch.dtype, m: int, n: int, k: int, trans_a: bool, trans_b: bool, alpha: float
) -> Tuple[str, ModuleType]:
    func_name = f"mm_{_gen_mm_key(dtype, m, n, k, trans_a, trans_b, alpha)}"
    kwargs = _mm_configs(dtype, m, n, k, trans_a, trans_b, alpha, func_name)
    src_code = _MM_TEMPLATE.format(**kwargs)
    if _DEBUG_MODE:
        os.makedirs(os.path.dirname("triton_debug/"), exist_ok=True)
        with open(f"triton_debug/{func_name}.py", "w") as f:
            f.write(src_code)
    return func_name, PyCodeCache().load(src_code)


def _gen_gemm_key(
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    stride_cm: int,
    stride_cn: int,
    trans_a: bool,
    trans_b: bool,
    alpha: float,
    beta: float,
) -> int:
    return hash(f"gemm|{dtype}|{m}|{n}|{k}|{stride_cm}|{stride_cn}|{trans_a}|{trans_b}|{alpha}|{beta}") % (10**8)


def _gen_gemm_module(
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    stride_cm: int,
    stride_cn: int,
    trans_a: bool,
    trans_b: bool,
    alpha: float,
    beta: float,
) -> Tuple[str, ModuleType]:
    func_name = f"gemm_{_gen_gemm_key(dtype, m, n, k, stride_cm, stride_cn, trans_a, trans_b, alpha, beta)}"
    kwargs = _mm_configs(dtype, m, n, k, trans_a, trans_b, alpha, func_name)
    kwargs["stride_cm"] = stride_cm
    kwargs["stride_cn"] = stride_cn
    kwargs["beta"] = beta
    src_code = _GEMM_TEMPLATE.format(**kwargs)
    if _DEBUG_MODE:
        os.makedirs(os.path.dirname("triton_debug/"), exist_ok=True)
        with open(f"triton_debug/{func_name}.py", "w") as f:
            f.write(src_code)
    return func_name, PyCodeCache().load(src_code)


def _gen_bmm_key(
    dtype: torch.dtype, m: int, n: int, k: int, batch_a: int, batch_b: int, trans_a: bool, trans_b: bool, alpha: float
) -> int:
    return hash(f"bmm|{dtype}|{m}|{n}|{k}|{batch_a}|{batch_b}|{trans_a}|{trans_b}|{alpha}") % (10**8)


def _gen_bmm_module(
    dtype: torch.dtype, m: int, n: int, k: int, batch_a: int, batch_b: int, trans_a: bool, trans_b: bool, alpha: float
) -> Tuple[str, ModuleType]:
    func_name = f"bmm_{_gen_bmm_key(dtype, m, n, k, batch_a, batch_b, trans_a, trans_b, alpha)}"
    kwargs = _mm_configs(dtype, m, n, k, trans_a, trans_b, alpha, func_name)
    batch = batch_a if batch_a >= batch_b else batch_b
    kwargs["stride_aq"] = m * k if batch_a == batch else 0
    kwargs["stride_bq"] = k * n if batch_b == batch else 0
    kwargs["batch"] = batch
    src_code = _BMM_TEMPLATE.format(**kwargs)
    if _DEBUG_MODE:
        os.makedirs(os.path.dirname("triton_debug/"), exist_ok=True)
        with open(f"triton_debug/{func_name}.py", "w") as f:
            f.write(src_code)
    return func_name, PyCodeCache().load(src_code)


def _matmul_internal(a, b, out, **kwargs):
    rank_a = len(a.shape)
    rank_b = len(b.shape)
    assert rank_a >= 2 and rank_b >= 2
    trans_a = kwargs.get("trans_a", False)
    trans_b = kwargs.get("trans_b", False)
    alpha = kwargs.get("alpha", 1.0)
    m = a.shape[-1] if trans_a else a.shape[-2]
    k = a.shape[-2] if trans_a else a.shape[-1]
    bk = b.shape[-1] if trans_b else b.shape[-2]
    assert k == bk
    n = b.shape[-2] if trans_b else b.shape[-1]
    assert rank_a == 2 or rank_b == 2 or a.shape[:-2] == b.shape[:-2]
    batch_shape = a.shape[:-2] if rank_a >= 3 else b.shape[:-2]
    if out is None:
        out = torch.empty((*batch_shape, m, n), dtype=a.dtype, device=a.device)
    else:
        assert out.shape == (*batch_shape, m, n)
    batch = math.prod(batch_shape)
    dtype = a.dtype
    if batch == 1 or (rank_a >= 3 and not trans_a and rank_b == 2):
        if batch != 1:
            m = batch * m
        func_name, mod = ModuleCache.load(_gen_mm_key, _gen_mm_module, dtype, m, n, k, trans_a, trans_b, alpha)
    else:
        batch_a = batch if rank_a >= 3 else 1
        batch_b = batch if rank_b >= 3 else 1
        func_name, mod = ModuleCache.load(
            _gen_bmm_key, _gen_bmm_module, dtype, m, n, k, batch_a, batch_b, trans_a, trans_b, alpha
        )
    func = getattr(mod, func_name)
    func(a, b, out)
    return out


def triton_matmul(a, b, **kwargs):
    """
    Compute matrix multiplication of two tensors.
    If trans_a is True in kwargs, input a will be transposed on last two dimensions before multiplication.
    If trans_b is True in kwargs, input b will be transposed on last two dimensions before multiplication.
    If alpha is specified in kwargs, the product will be multiplied by alpha.
    The input tensors can be of rank 2 or higher, but currently support only simple broadcasting on batch dimensions.
    I.e., the batch dimensions of a and b must either be equal, or one of them must be 1.
    """

    return _matmul_internal(a, b, None, **kwargs)


def triton_matmul_out(a, b, out, **kwargs):
    """
    Same as triton_matmul, except that the output is allocated and passed from outside.
    """

    _matmul_internal(a, b, out, **kwargs)


def _gemm_internal(a, b, c, out, **kwargs):
    assert len(a.shape) == 2 and len(b.shape) == 2
    trans_a = kwargs.get("trans_a", False)
    trans_b = kwargs.get("trans_b", False)
    alpha = kwargs.get("alpha", 1.0)
    beta = kwargs.get("beta", 1.0)
    m = a.shape[-1] if trans_a else a.shape[-2]
    k = a.shape[-2] if trans_a else a.shape[-1]
    bk = b.shape[-1] if trans_b else b.shape[-2]
    assert k == bk
    n = b.shape[-2] if trans_b else b.shape[-1]
    stride_cm = c.shape[-1] if len(c.shape) == 2 and c.shape[-2] == m else 0
    stride_cn = 1 if len(c.shape) >= 1 and c.shape[-1] == n else 0
    if out is None:
        out = torch.empty(m, n, dtype=a.dtype, device=a.device)
    else:
        assert out.shape == (m, n)
    dtype = a.dtype
    func_name, mod = ModuleCache.load(
        _gen_gemm_key, _gen_gemm_module, dtype, m, n, k, stride_cm, stride_cn, trans_a, trans_b, alpha, beta
    )
    func = getattr(mod, func_name)
    func(a, b, c, out)
    return out


def triton_gemm(a, b, c, **kwargs):
    """
    Compute alpha * a @ b + beta * c. Here a and b are 2D tensors, and c is broadcastable to the result.
    If trans_a is True in kwargs, input a will be transposed before multiplication.
    If trans_b is True in kwargs, input b will be transposed before multiplication.
    """

    return _gemm_internal(a, b, c, None, **kwargs)


def triton_gemm_out(a, b, c, out, **kwargs):
    """
    Same as triton_gemm, except that the output is allocated and passed from outside.
    """

    _gemm_internal(a, b, c, out, **kwargs)
