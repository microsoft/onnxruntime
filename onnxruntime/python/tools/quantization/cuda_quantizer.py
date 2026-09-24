# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""CUDA weight-only quantization helpers.

This module contains small Python utilities for producing the weight layouts
consumed by CUDA weight-only kernels. The blockwise quantizers wrap the same C++
pybind entry points used by runtime prepacking, and the mixed-GEMM weight packer
is a PyTorch reimplementation of the runtime CUDA packing, so tests and model
builders can generate byte-identical quantized weights. The PyTorch packer runs
on CUDA when a device is available and falls back to CPU otherwise, which is the
only option on platforms where the standalone CUDA packer is not built (Windows).
A GPU-gated parity test validates it against that standalone CUDA packer.

Two storage families are exposed:

* raw MatMulNBits blockwise storage, laid out by output channel as ``[N, K/pack]``;
* CUDA mixed-GEMM prepacked storage. MatMulNBits prepacked initializers keep the
    schema shape ``[N, K/block_size, block_size*bits/8]`` and require the node
    attribute ``weight_prepacked=1``. QMoE/CUTLASS callers use the kernel-facing
    shape ``[K, N/pack]``.

All public helpers take one logical expert weight matrix with shape ``[N, K]``.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import numpy as np

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch


def _get_torch():
    """Import torch lazily so importing onnxruntime.quantization does not require torch."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("CUDA weight-only quantization requires torch. Please install torch to use it.") from e

    return torch


def _get_pack_weights_for_cuda_mixed_gemm():
    """Return the standalone CUDA mixed-GEMM weight packer (parity oracle).

    Production packing uses the PyTorch implementation (``_pack_weights_for_cuda_mixed_gemm``).
    This standalone packer lives in ``onnxruntime.capi.onnxruntime_cuda_quant_preprocess``, a
    separate extension module that links the CUDA runtime (built only on non-Windows CUDA
    builds). It is imported lazily here (never at ``import onnxruntime`` time) and is used by
    the parity test to validate the PyTorch packer byte-for-byte.
    """
    try:
        from onnxruntime.capi import onnxruntime_cuda_quant_preprocess as _cuda_quant  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The standalone CUDA weight packer (onnxruntime_cuda_quant_preprocess) is unavailable; "
            "it is built only on non-Windows onnxruntime-gpu CUDA builds."
        ) from e

    try:
        return _cuda_quant.pack_weights_for_cuda_mixed_gemm
    except AttributeError as e:
        raise ImportError("onnxruntime_cuda_quant_preprocess is missing pack_weights_for_cuda_mixed_gemm.") from e


def has_cuda_weight_prepacking() -> bool:
    """Return True if mixed-GEMM weight prepacking is available.

    Prepacking is implemented with PyTorch (CUDA when available, CPU otherwise), so it is
    available whenever torch is importable. Callers use this to skip prepack code paths
    (and tests) when torch is unavailable.
    """
    try:
        _get_torch()
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def _warn_cpu_prepack_once() -> None:
    _logger.warning(
        "CUDA device is not available; packing mixed-GEMM weights on CPU with PyTorch. "
        "This is correct but significantly slower for large Mixture-of-Experts models. "
        "Pack on a CUDA-enabled machine for best performance."
    )


def _prepack_device():
    """Pick the torch device for mixed-GEMM weight packing (CUDA if available, else CPU)."""
    torch = _get_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    _warn_cpu_prepack_once()
    return torch.device("cpu")


def _preprocess_weights_for_mixed_gemm_torch(tensor, bits: int, sm: int):
    """PyTorch port of the runtime CUDA ``preprocess_weights_for_mixed_gemm``.

    ``tensor`` is a signed int8 weight in ``(K, N/pack)`` packed row-major layout on any
    device. Returns the CUTLASS mixed-GEMM layout with the same shape/dtype/device. This
    mirrors ``preprocess_weights_for_mixed_gemm_cuda`` (permute_B_rows -> subbyte_transpose
    -> interleave_column_major -> add_bias_and_interleave) so its output is byte-identical
    to the standalone CUDA packer, for both the SM80 (Ampere) and SM90 (Hopper) layouts.
    """
    torch = _get_torch()
    bits_a = 16  # fp16/bf16 activations
    bits_b = 4 if bits == 4 else 8

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    permutation_map = {
        "16_8": [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
        "16_4": [
            0,
            1,
            8,
            9,
            16,
            17,
            24,
            25,
            2,
            3,
            10,
            11,
            18,
            19,
            26,
            27,
            4,
            5,
            12,
            13,
            20,
            21,
            28,
            29,
            6,
            7,
            14,
            15,
            22,
            23,
            30,
            31,
        ],
    }
    mma_shape_n = 8
    b_rows_per_mma = 8 * 16 // bits_b

    num_experts, num_rows, num_cols = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    if num_rows % b_rows_per_mma != 0 or num_cols % mma_shape_n != 0:
        raise ValueError(
            f"weight shape (rows={num_rows}, packed_cols={num_cols}) is incompatible with mixed-GEMM "
            f"packing (rows must be a multiple of {b_rows_per_mma}, packed cols a multiple of {mma_shape_n})."
        )

    # permute_B_rows_for_mixed_gemm
    if sm < 100:
        pmap = permutation_map[f"{bits_a}_{bits_b}"]
        row_idx = [(r // b_rows_per_mma) * b_rows_per_mma + pmap[r % b_rows_per_mma] for r in range(num_rows)]
        tensor = tensor[:, row_idx, :]

    # subbyte_transpose
    original_shape = tensor.shape
    if bits_b == 4:
        u = tensor.view(torch.uint8)
        high = (u >> 4).permute(0, 2, 1).unsqueeze(2)
        low = ((u << 4) >> 4).permute(0, 2, 1).unsqueeze(2)
        merged = torch.cat([low, high], dim=2).reshape(u.shape[0], -1, u.shape[1])
        merged = merged[:, :, 0::2] + merged[:, :, 1::2] * 16
        tensor = merged.view(torch.int8).reshape(original_shape)
    else:
        tensor = tensor.permute(0, 2, 1).reshape(original_shape)

    # interleave_column_major_tensor
    interleave = bits_a // bits_b
    if interleave > 1 and sm < 90:
        rows_per_tile = 128 * 8 // bits_a
        elts_in_int32 = 32 // bits_b
        if num_rows % elts_in_int32 != 0 or num_rows % rows_per_tile != 0:
            raise ValueError(f"num_rows ({num_rows}) is incompatible with column-interleave tiling.")
        tensor = tensor.reshape(
            num_experts, -1, interleave, num_rows // rows_per_tile, rows_per_tile * 4 // elts_in_int32
        )
        tensor = tensor.permute(0, 1, 3, 2, 4).reshape(original_shape)

    # add_bias_and_interleave_quantized_tensor_inplace
    if bits_b == 8:
        t = tensor.to(torch.int64)  # widen so the +128 rebias cannot overflow int8
        t += -256 * (t > 127).to(torch.int64) + 128
        t = t.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(original_shape)
        tensor = t.to(torch.uint8).view(torch.int8)
    else:
        u = tensor.view(torch.uint8)
        high = (u >> 4).unsqueeze(-1)
        low = ((u << 4) >> 4).unsqueeze(-1)
        merged = torch.cat([low, high], dim=-1).reshape(u.shape[0], u.shape[1], -1)
        merged = merged.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(merged.shape)
        merged = merged.to(torch.int16)
        merged += -16 * (merged > 7).to(torch.int16) + 8
        merged = merged[:, :, 0::2] + merged[:, :, 1::2] * 16
        tensor = merged.to(torch.uint8).view(torch.int8)

    return tensor.squeeze(0).contiguous()


def _pack_weights_for_cuda_mixed_gemm(q_weights, n: int, k: int, bits: int, force_arch: int = 80) -> np.ndarray:
    """PyTorch implementation of the CUDA ``pack_weights_for_cuda_mixed_gemm``.

    ``q_weights`` is ORT's unsigned MatMulNBits/QMoE storage ``(N, K/pack)`` (uint8). Returns
    a flat ``int8`` numpy array with the CUTLASS mixed-GEMM layout, byte-identical to the
    standalone CUDA packer. Runs on CUDA when available, otherwise on CPU.
    """
    torch = _get_torch()
    bits = int(bits)
    force_arch = int(force_arch)
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}.")
    if force_arch not in (80, 90):
        raise ValueError(f"force_arch must be 80 (SM80) or 90 (SM90), got {force_arch}.")
    pack = 8 // bits
    device = _prepack_device()

    q = torch.as_tensor(np.ascontiguousarray(q_weights)).view(torch.uint8).reshape(n, k // pack).to(device)

    # Front-end adaptor: transpose ORT (N, K) -> (K, N) and convert unsigned -> signed int8.
    if bits == 4:
        low = (q & 0x0F).to(torch.int16)
        high = (q >> 4).to(torch.int16)
        unpacked = torch.empty((n, k), dtype=torch.int16, device=device)
        unpacked[:, 0::2] = low
        unpacked[:, 1::2] = high
        signed_t = (unpacked - 8).transpose(0, 1).contiguous()  # (K, N), zero point 8
        packed_t = ((signed_t[:, 0::2] & 0x0F) | ((signed_t[:, 1::2] & 0x0F) << 4)).to(torch.uint8).view(torch.int8)
    else:
        signed_t = (q.to(torch.int16) - 128).transpose(0, 1).contiguous()  # (K, N), zero point 128
        packed_t = signed_t.to(torch.uint8).view(torch.int8)

    out = _preprocess_weights_for_mixed_gemm_torch(packed_t.contiguous(), bits, force_arch)
    return out.reshape(-1).cpu().numpy()


def _get_quantize_matmul_nbits():
    """Return MatMulNBits blockwise quantizers from the ORT pybind module."""
    try:
        from onnxruntime.capi._pybind_state import (  # noqa: PLC0415
            quantize_matmul_4bits,
            quantize_matmul_8bits,
        )
    except ImportError as e:
        raise ImportError(
            "CUDA blockwise quantization requires quantize_matmul_4bits and quantize_matmul_8bits from onnxruntime."
        ) from e

    return quantize_matmul_4bits, quantize_matmul_8bits


class CudaQuantizer:
    """CUDA quantizer utilities for MoE/QMoE and MatMulNBits-style weight-only kernels.

    The methods are stateless; callers may use the class directly without
    constructing an object.
    """

    @staticmethod
    def qmoe_symmetric_per_channel_quantize(
        weights: torch.Tensor,
        bits: int,
        *,
        unsigned_full_range: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize one QMoE expert with symmetric per-channel storage.

        ``weights`` has logical shape ``[N, K]``. Returns raw QMoE storage
        ``[N, K/pack]`` and scales ``[N]``. By default, this emits the ORT CUDA
        QMoE storage contract: unsigned bytes/nibbles with an implicit zero-point
        offset, so each stored value is ``q + zero_point`` even though the numeric
        quantization is symmetric. By default it uses the full ``[-8, 7]`` /
        ``[-128, 127]`` range. Set ``unsigned_full_range=False`` to use the legacy
        ``[-7, 7]`` / ``[-127, 127]`` range.
        """
        torch = _get_torch()

        weights = weights.detach().cpu().to(torch.float32).contiguous()
        bits = int(bits)
        if bits not in (4, 8):
            raise ValueError(f"QMoE per-channel quantization only supports 4 or 8 bits, got {bits}.")

        n, k = weights.shape
        pack = 8 // bits
        if k % pack != 0:
            raise ValueError(f"K ({k}) must be divisible by {pack} for QMoE per-channel quantization.")

        if bits == 4:
            if unsigned_full_range:
                qmin, qmax, scale_divisor, zero_point = -8, 7, 8, 8
            else:
                qmin, qmax, scale_divisor, zero_point = -7, 7, 7, 8
        else:  # bits == 8, already validated above
            if unsigned_full_range:
                qmin, qmax, scale_divisor, zero_point = -128, 127, 128, 128
            else:
                qmin, qmax, scale_divisor, zero_point = -127, 127, 127, 128
        scales = weights.abs().amax(dim=1, keepdim=True) / float(scale_divisor)
        scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
        quantized = torch.clamp(torch.round(weights / scales), qmin, qmax).to(torch.int16).contiguous()
        quantized = (quantized + zero_point).to(torch.uint8)

        if bits == 4:
            qweight = (quantized[:, 0::2] & 0xF) | ((quantized[:, 1::2] & 0xF) << 4)
            qweight = qweight.to(torch.uint8)
        else:
            qweight = quantized

        return qweight.contiguous(), scales.squeeze(-1).contiguous()

    @staticmethod
    def qmoe_per_channel_quantize(
        weights: torch.Tensor,
        bits: int,
        prepack: bool,
        force_arch: int = 80,
        *,
        unsigned_full_range: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize per-channel QMoE weights and optionally CUTLASS-prepack them.

        When ``prepack`` is true, returned weights have shape ``[K, N/pack]``.
        Otherwise, returned weights keep raw per-channel storage ``[N, K/pack]``.
        Prepacking uses PyTorch (CUDA when available, CPU otherwise).
        """
        torch = _get_torch()

        qweight, scales = CudaQuantizer.qmoe_symmetric_per_channel_quantize(
            weights,
            bits,
            unsigned_full_range=unsigned_full_range,
        )
        if not prepack:
            return qweight, scales

        n, k = weights.shape
        pack = 8 // int(bits)
        if n % pack != 0:
            raise ValueError(f"N ({n}) must be divisible by {pack} for CUDA QMoE prepacked weights.")

        packed = _pack_weights_for_cuda_mixed_gemm(qweight.numpy(), n, k, int(bits), force_arch)
        packed = np.asarray(packed).view(np.uint8).reshape(k, n // pack)
        return torch.from_numpy(np.ascontiguousarray(packed)), scales

    @staticmethod
    def _matmulnbits_blockwise_quantize_impl(
        weights: torch.Tensor,
        bits: int,
        block_size: int,
        *,
        symmetric: bool,
        abs_scales: bool,
        unsigned_full_range: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``weights`` with MatMulNBits pybinds and return unflattened storage."""
        torch = _get_torch()

        bits = int(bits)
        block_size = int(block_size)
        w = weights.detach().cpu().to(torch.float32).contiguous().numpy()
        n, k = w.shape
        if bits not in (4, 8):
            raise ValueError(f"Blockwise quantization only supports 4 or 8 bits, got {bits}.")
        if block_size <= 0:
            raise ValueError(f"Blockwise quantization requires a positive block_size, got {block_size}.")

        num_blocks = (k + block_size - 1) // block_size
        pack = 8 // bits
        blob_size = (block_size + pack - 1) // pack

        if symmetric:
            if bits == 4:
                qmin, qmax, scale_divisor, zero_point = (-8, 7, 8, 8) if unsigned_full_range else (-7, 7, 7, 8)
            else:
                qmin, qmax, scale_divisor, zero_point = (
                    (-128, 127, 128, 128) if unsigned_full_range else (-127, 127, 127, 128)
                )

            padded_k = num_blocks * block_size
            if padded_k != k:
                w = np.pad(w, ((0, 0), (0, padded_k - k)), "constant")

            blocked = w.reshape(n, num_blocks, block_size)
            scales = np.max(np.abs(blocked), axis=2).astype(np.float32) / np.float32(scale_divisor)
            scales = np.maximum(scales, np.finfo(np.float32).eps)
            quantized = np.clip(np.rint(blocked / scales[:, :, np.newaxis]), qmin, qmax).astype(np.int16)
            quantized = (quantized + zero_point).astype(np.uint8)

            if bits == 4:
                qweight = np.zeros((n, num_blocks, blob_size), dtype=np.uint8)
                qweight[:, :, : quantized[:, :, 0::2].shape[2]] = quantized[:, :, 0::2] & 0xF
                qweight[:, :, : quantized[:, :, 1::2].shape[2]] |= (quantized[:, :, 1::2] & 0xF) << 4
            else:
                qweight = quantized

            zero_points = np.zeros((n, (num_blocks + 1) // 2 if bits == 4 else num_blocks), dtype=np.uint8)
            return torch.from_numpy(qweight), torch.from_numpy(scales), torch.from_numpy(zero_points)

        w_t = np.ascontiguousarray(w.T)
        qweight = np.zeros((n, num_blocks, blob_size), dtype=np.uint8)
        scales = np.zeros((n, num_blocks), dtype=np.float32)
        zero_points = np.zeros((n, (num_blocks + 1) // 2 if bits == 4 else num_blocks), dtype=np.uint8)

        quantize_matmul_4bits, quantize_matmul_8bits = _get_quantize_matmul_nbits()
        quantize = quantize_matmul_4bits if bits == 4 else quantize_matmul_8bits
        quantize(qweight, w_t, scales, zero_points, block_size, n, k, symmetric)

        if abs_scales:
            scales = np.abs(scales)

        return torch.from_numpy(qweight), torch.from_numpy(scales), torch.from_numpy(zero_points)

    @staticmethod
    def matmulnbits_blockwise_quantize(
        weights: torch.Tensor,
        bits: int,
        block_size: int,
        *,
        symmetric: bool = True,
        return_zero_points: bool = False,
        abs_scales: bool = True,
        flatten_qweight: bool = True,
        unsigned_full_range: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize one expert with ONNX Runtime's MatMulNBits blockwise encoding.

        ``weights`` has logical shape ``[N, K]``. Returns raw flattened storage
        ``[N, ceil(K/block_size)*ceil(block_size/pack)]`` and block scales
        ``[N, ceil(K/block_size)]`` by default. Set ``flatten_qweight=False`` for
        the MatMulNBits initializer shape
        ``[N, ceil(K/block_size), ceil(block_size/pack)]``.
        Set ``return_zero_points=True`` to also return packed block zero-points.
        Symmetric quantization uses the full ``[-8, 7]`` / ``[-128, 127]`` range
        by default. Set ``unsigned_full_range=False`` to use the legacy
        ``[-7, 7]`` / ``[-127, 127]`` range.
        """
        qweight, scales, zero_points = CudaQuantizer._matmulnbits_blockwise_quantize_impl(
            weights,
            bits,
            block_size,
            symmetric=symmetric,
            abs_scales=abs_scales,
            unsigned_full_range=unsigned_full_range,
        )
        if flatten_qweight:
            qweight = qweight.reshape(qweight.shape[0], -1).contiguous()
        if return_zero_points:
            return qweight, scales, zero_points

        return qweight, scales

    @staticmethod
    def matmulnbits_prepacked_blockwise_quantize(
        weights: torch.Tensor,
        bits: int,
        block_size: int,
        force_arch: int = 80,
        *,
        symmetric: bool = True,
        return_zero_points: bool = False,
        abs_scales: bool = True,
        unsigned_full_range: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize and CUDA-prepack one MatMulNBits weight initializer.

        ``weights`` has logical shape ``[N, K]``. Returns ``B`` with the standard
        MatMulNBits initializer shape ``[N, K/block_size, block_size*bits/8]``
        and scales with shape ``[N, K/block_size]``.

        The ``force_arch`` value selects the mixed-GEMM weight layout and must match
        the ``weight_prepacked`` attribute set on the MatMulNBits node:

        * ``force_arch=80`` (default): SM80/Ampere layout, consumed by the SM80 kernel
          (also used on newer GPUs via the compatibility path). Use ``weight_prepacked=1``.
        * ``force_arch=90``: SM90/Hopper layout, consumed by the native SM90 TMA/WGMMA
          kernel. Use ``weight_prepacked=2``. Requires ``block_size`` in {64, 128}.
        """
        torch = _get_torch()

        bits = int(bits)
        block_size = int(block_size)
        force_arch = int(force_arch)
        if force_arch not in (80, 90):
            raise ValueError(f"force_arch must be 80 (SM80) or 90 (SM90), but got {force_arch}.")
        # The native SM90 kernel needs group_size to be a multiple of the 64-element Hopper K tile,
        # so block_size=32 is only supported by the SM80/Ampere-class kernel.
        allowed_block_sizes = (32, 64, 128) if force_arch == 80 else (64, 128)
        if block_size not in allowed_block_sizes:
            raise ValueError(
                f"block_size must be one of {allowed_block_sizes} for force_arch={force_arch}, but got {block_size}."
            )
        n, k = weights.shape
        if k % block_size != 0:
            raise ValueError(f"K ({k}) must be divisible by block_size ({block_size}) for CUDA-prepacked weights.")

        qweight, scales, zero_points = CudaQuantizer._matmulnbits_blockwise_quantize_impl(
            weights,
            bits,
            block_size,
            symmetric=symmetric,
            abs_scales=abs_scales,
            unsigned_full_range=unsigned_full_range,
        )

        pack_weights_for_cuda_mixed_gemm = _pack_weights_for_cuda_mixed_gemm
        packed = pack_weights_for_cuda_mixed_gemm(qweight.reshape(n, -1).numpy(), n, k, bits, force_arch)
        packed = np.asarray(packed).view(np.uint8).reshape(qweight.shape)
        packed = torch.from_numpy(np.ascontiguousarray(packed))
        if return_zero_points:
            return packed, scales, zero_points

        return packed, scales

    @staticmethod
    def qmoe_prepacked_blockwise_quantize(
        weights: torch.Tensor,
        bits: int,
        block_size: int,
        force_arch: int = 80,
        *,
        symmetric: bool = True,
        return_zero_points: bool = False,
        abs_scales: bool = False,
        unsigned_full_range: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize one expert and CUTLASS-prepack it for CUDA QMoE fpA_intB GEMM.

        ``weights`` has logical shape ``[N, K]``. Returns ``qweight`` with shape
        ``[K, N/pack]`` and block scales with shape ``[N, K/block_size]``.
        Set ``return_zero_points=True`` to also return packed block zero-points.
        """
        bits = int(bits)
        block_size = int(block_size)
        n, k = weights.shape
        pack = 8 // bits
        if k % block_size != 0:
            raise ValueError(f"K ({k}) must be divisible by block_size ({block_size}) for CUDA-prepacked weights.")
        if n % pack != 0:
            raise ValueError(f"N ({n}) must be divisible by {pack} for QMoE blockwise quantization.")

        qweight, scales, zero_points = CudaQuantizer._matmulnbits_blockwise_quantize_impl(
            weights,
            bits,
            block_size,
            symmetric=symmetric,
            abs_scales=abs_scales,
            unsigned_full_range=unsigned_full_range,
        )

        pack_weights_for_cuda_mixed_gemm = _pack_weights_for_cuda_mixed_gemm
        packed = pack_weights_for_cuda_mixed_gemm(qweight.reshape(n, -1).numpy(), n, k, bits, force_arch)
        packed = np.asarray(packed).view(np.uint8).reshape(k, n // pack)
        torch = _get_torch()
        packed = torch.from_numpy(np.ascontiguousarray(packed))
        if return_zero_points:
            return packed, scales, zero_points

        return packed, scales

    @staticmethod
    def symmetric_blockwise_quantize(
        weights: torch.Tensor,
        bits: int,
        block_size: int,
        *,
        unsigned_full_range: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize one expert with a pure-PyTorch symmetric blockwise encoding.

        This helper is useful for non-CUDA reference paths. Unlike the pybind-backed
        helpers above, it pads the last dimension when it is not divisible by
        ``block_size`` and returns storage with the same leading shape as ``weights``.
        """
        torch = _get_torch()

        weights = weights.detach().cpu().contiguous()
        original_shape = weights.shape
        bits = int(bits)
        block_size = int(block_size)
        if bits == 4:
            qmin, qmax, scale_divisor = (-8, 7, 8) if unsigned_full_range else (-7, 7, 7)
        elif bits == 8:
            qmin, qmax, scale_divisor = (-128, 127, 128) if unsigned_full_range else (-127, 127, 127)
        else:
            raise ValueError(f"CUDA blockwise quantization only supports 4 or 8 bits, got {bits}.")

        last_dim = original_shape[-1]
        num_blocks = (last_dim + block_size - 1) // block_size
        pad_size = num_blocks * block_size - last_dim
        if pad_size > 0:
            pad_shape = list(original_shape)
            pad_shape[-1] = pad_size
            padding = torch.zeros(pad_shape, dtype=weights.dtype, device=weights.device)
            weights_padded = torch.cat([weights, padding], dim=-1)
        else:
            weights_padded = weights

        reshaped_weights = weights_padded.view(*original_shape[:-1], num_blocks, block_size)
        block_max_abs = torch.max(torch.abs(reshaped_weights), dim=-1)[0]
        scales = torch.clamp(block_max_abs / scale_divisor, min=1e-8)

        quantized = torch.round(reshaped_weights / scales.unsqueeze(-1))
        quantized = torch.clamp(quantized, qmin, qmax)

        if bits == 4:
            quantized_flat = quantized.to(torch.int8).view(*original_shape[:-1], num_blocks * block_size)
            if pad_size > 0:
                quantized_flat = quantized_flat[..., :-pad_size]

            quantized_uint4 = (quantized_flat + 8).to(torch.uint8)
            packed_shape = list(original_shape)
            packed_shape[-1] = (original_shape[-1] + 1) // 2
            qweight = torch.zeros(packed_shape, dtype=torch.uint8, device=weights.device)
            qweight[..., :] = quantized_uint4[..., 0::2] & 0xF
            if quantized_uint4.shape[-1] > 1:
                qweight[..., : quantized_uint4[..., 1::2].shape[-1]] |= (quantized_uint4[..., 1::2] & 0xF) << 4
        else:
            qweight = quantized.to(torch.int8).view(*original_shape[:-1], num_blocks * block_size)
            if pad_size > 0:
                qweight = qweight[..., :-pad_size]
            qweight = qweight.view(original_shape)

        return qweight.cpu(), scales.cpu()
