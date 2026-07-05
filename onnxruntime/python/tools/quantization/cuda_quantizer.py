# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""CUDA weight-only quantization helpers.

This module contains small Python utilities for producing the weight layouts
consumed by CUDA weight-only kernels. The helpers deliberately wrap the same C++
pybind entry points used by runtime prepacking so tests and model builders can
generate byte-identical quantized weights.

Two storage families are exposed:

* raw MatMulNBits blockwise storage, laid out by output channel as ``[N, K/pack]``;
* CUDA mixed-GEMM prepacked storage. MatMulNBits prepacked initializers keep the
    schema shape ``[N, K/block_size, block_size*bits/8]`` and require the node
    attribute ``weight_prepacked=1``. QMoE/CUTLASS callers use the kernel-facing
    shape ``[K, N/pack]``.

All public helpers take one logical expert weight matrix with shape ``[N, K]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
    """Return the CUDA mixed-GEMM weight prepacker from the ORT pybind module."""
    try:
        from onnxruntime.capi import _pybind_state as _pybind  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "CUDA weight prepacking requires pack_weights_for_cuda_mixed_gemm from an onnxruntime-gpu CUDA build."
        ) from e

    try:
        return _pybind.pack_weights_for_cuda_mixed_gemm
    except AttributeError as e:
        raise ImportError(
            "CUDA weight prepacking requires pack_weights_for_cuda_mixed_gemm from an onnxruntime-gpu CUDA build."
        ) from e


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
    def symmetric_per_channel_quantize(
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
        n, k = weights.shape
        pack = 8 // bits
        if k % pack != 0:
            raise ValueError(f"K ({k}) must be divisible by {pack} for QMoE per-channel quantization.")

        if bits == 4:
            if unsigned_full_range:
                qmin, qmax, scale_divisor, zero_point = -8, 7, 8, 8
            else:
                qmin, qmax, scale_divisor, zero_point = -7, 7, 7, 8
        elif bits == 8:
            if unsigned_full_range:
                qmin, qmax, scale_divisor, zero_point = -128, 127, 128, 128
            else:
                qmin, qmax, scale_divisor, zero_point = -127, 127, 127, 128
        else:
            raise ValueError(f"QMoE per-channel quantization only supports 4 or 8 bits, got {bits}.")

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
    def cuda_per_channel_quantize(
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
        CUDA prepacking requires ``pack_weights_for_cuda_mixed_gemm`` from an
        onnxruntime-gpu CUDA build.
        """
        torch = _get_torch()

        qweight, scales = CudaQuantizer.symmetric_per_channel_quantize(
            weights,
            bits,
            unsigned_full_range=unsigned_full_range,
        )
        if not prepack:
            return qweight, scales

        pack_weights_for_cuda_mixed_gemm = _get_pack_weights_for_cuda_mixed_gemm()

        n, k = weights.shape
        pack = 8 // int(bits)
        if n % pack != 0:
            raise ValueError(f"N ({n}) must be divisible by {pack} for CUDA QMoE prepacked weights.")

        packed = pack_weights_for_cuda_mixed_gemm(qweight.numpy(), n, k, int(bits), force_arch)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``weights`` with MatMulNBits pybinds and return unflattened storage."""
        torch = _get_torch()

        bits = int(bits)
        block_size = int(block_size)
        w = weights.detach().cpu().to(torch.float32).contiguous().numpy()
        n, k = w.shape
        if bits not in (4, 8):
            raise ValueError(f"CUDA blockwise quantization only supports 4 or 8 bits, got {bits}.")
        if k % block_size != 0:
            raise ValueError(f"K ({k}) must be divisible by block_size ({block_size}) for CUDA blockwise quantization.")

        num_blocks = k // block_size
        pack = 8 // bits

        w_t = np.ascontiguousarray(w.T)
        qweight = np.zeros((n, num_blocks, block_size // pack), dtype=np.uint8)
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
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize one expert with ONNX Runtime's MatMulNBits blockwise encoding.

        ``weights`` has logical shape ``[N, K]``. Returns raw MatMulNBits storage
        ``[N, K/pack]`` and block scales ``[N, K/block_size]`` by default.
        Set ``return_zero_points=True`` to also return packed block zero-points.
        """
        qweight, scales, zero_points = CudaQuantizer._matmulnbits_blockwise_quantize_impl(
            weights,
            bits,
            block_size,
            symmetric=symmetric,
            abs_scales=abs_scales,
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize and CUDA-prepack one MatMulNBits weight initializer.

        ``weights`` has logical shape ``[N, K]``. Returns ``B`` with the standard
        MatMulNBits initializer shape ``[N, K/block_size, block_size*bits/8]``
        and scales with shape ``[N, K/block_size]``. Use the returned ``B`` with
        ``weight_prepacked=1`` on the MatMulNBits node.

        The default ``force_arch=80`` matches the current CUDA MatMulNBits
        fpA_intB path: runtime prepacking and offline prepacking both consume the
        SM80 mixed-GEMM layout, including on newer GPUs.
        """
        torch = _get_torch()

        bits = int(bits)
        n, k = weights.shape
        qweight, scales, zero_points = CudaQuantizer._matmulnbits_blockwise_quantize_impl(
            weights,
            bits,
            block_size,
            symmetric=symmetric,
            abs_scales=abs_scales,
        )

        pack_weights_for_cuda_mixed_gemm = _get_pack_weights_for_cuda_mixed_gemm()
        packed = pack_weights_for_cuda_mixed_gemm(qweight.reshape(n, -1).numpy(), n, k, bits, force_arch)
        packed = np.asarray(packed).view(np.uint8).reshape(qweight.shape)
        packed = torch.from_numpy(np.ascontiguousarray(packed))
        if return_zero_points:
            return packed, scales, zero_points

        return packed, scales

    @staticmethod
    def cutlass_prepacked_blockwise_quantize(
        weights: torch.Tensor,
        bits: int,
        block_size: int,
        force_arch: int = 80,
        *,
        symmetric: bool = True,
        return_zero_points: bool = False,
        abs_scales: bool = False,
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
        if n % pack != 0:
            raise ValueError(f"N ({n}) must be divisible by {pack} for QMoE blockwise quantization.")

        qweight, scales, zero_points = CudaQuantizer._matmulnbits_blockwise_quantize_impl(
            weights,
            bits,
            block_size,
            symmetric=symmetric,
            abs_scales=abs_scales,
        )

        pack_weights_for_cuda_mixed_gemm = _get_pack_weights_for_cuda_mixed_gemm()
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
            qmin, qmax = -7, 7
        elif bits == 8:
            qmin, qmax = -127, 127
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
        scales = torch.clamp(block_max_abs / qmax, min=1e-8)

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
