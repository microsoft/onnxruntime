// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/prepacked_weights.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

class DynamicQuantMatMulFp8 final : public OpKernel {
 public:
  DynamicQuantMatMulFp8(const OpKernelInfo& info);

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_A_SCALE = 1,
    IN_A_ZERO_POINT = 2,
    IN_B = 3,
    IN_B_SCALE = 4,
    IN_B_ZERO_POINT = 5,
    IN_Y_SCALE = 6,
    IN_Y_ZERO_POINT = 7
  };

  enum OutputTensors : int { OUT_Y = 0 };

  static Status GetFp8Type(const Tensor& tensor, mlas_fp8_mode& out_type);
  static Status GetFp8Type(ONNX_NAMESPACE::TensorProto_DataType elem_type, mlas_fp8_mode& out_type);

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   gsl::span<const size_t> prepacked_buffer_sizes,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  static constexpr int GetBIdx() { return IN_B; }
  IAllocatorUniquePtr<void> quantized_b_;
  size_t quantized_b_size_{0};
  TensorShape b_shape_;
  mlas_fp8_mode b_type_{static_cast<mlas_fp8_mode>(0)};
  bool has_b_type_{false};
  size_t block_size_m_{128};
  size_t block_size_k_{128};
  size_t block_size_n_{128};
};

}  // namespace contrib
}  // namespace onnxruntime
