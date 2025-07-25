/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file device/quant_b4_gemm.h
 * @brief Launcher for fused GEMM kernel for fp16 x int4, where B matrix is blockwise quantized to 4bits.
 *
 * This is a competitor implementation of cutlass_ext/q4gemm/device/quantb_gemm.h. This one
 * is not based on cutlass.  Currently, this implementation performs better in smaller models
 * with batch size <= 16.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#include "gemm/kernel/quant_b4_gemm.h"

namespace mickey {
namespace gemm {
namespace device {
/**
 * @brief Kernel launcher for quantized GEMM with B matrix quantized to 4bits.
 */
template <
    typename QuantBlocking_,  ///! Shape of the quantization block, either 1xb or bx1
    bool has_quant_offset,    ///! Whether to use quantization offset
    typename WarpShape_,      ///! Warp-scoped matrix multiply-accumulate
    int SplitKSerial_ = 1,    ///! How many warps to split the K dimension in the same MxN block
    int Stages_ = 3           ///! Stages of the pipelined mainloop
    >
class QuantB4Gemm {
 public:
  using QuantBlocking = QuantBlocking_;
  using WarpShape = WarpShape_;
  static const int kSplitK = SplitKSerial_;
  static const int kStages = Stages_;

  using Kernel = mickey::gemm::kernel::QuantB4Gemm<QuantBlocking, has_quant_offset, WarpShape, kSplitK, kStages>;
  using Args = typename Kernel::Params;

  static cutlass::Status run(
      cudaStream_t stream,
      cutlass::gemm::GemmCoord const& problem_size,
      void* ptr_output,
      size_t output_byte_stride,
      void const* ptr_a,
      size_t a_byte_stride,
      void const* ptr_packed_b,
      size_t b_byte_stride,
      void const* ptr_scales,
      size_t scales_byte_stride,
      void const* ptr_zp = nullptr,
      size_t zp_byte_stride = 0) {
    Args args(problem_size, ptr_output, output_byte_stride,
              ptr_a, a_byte_stride, ptr_packed_b, b_byte_stride,
              ptr_scales, scales_byte_stride,
              ptr_zp, zp_byte_stride);
    cutlass::Status status = Kernel::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    dim3 grid(args.grid_tiled_shape_.m(), args.grid_tiled_shape_.n(), args.grid_tiled_shape_.k());
    dim3 block(Kernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename Kernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        std::cerr << "Failed to obtain maximum shared memory size " << smem_size << " for kernel: "
                  << cudaGetErrorString(result) << "\n";
        return cutlass::Status::kErrorInternal;
      }
    }

    cutlass::Kernel<Kernel><<<grid, block, smem_size, stream>>>(args);

    return cutlass::Status::kSuccess;
  }
};

}  // namespace device
}  // namespace gemm
}  // namespace mickey
