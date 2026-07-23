// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/prepacked_weights.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace contrib {

enum class Fp8Mode : int {
  E4M3Inf = 0,
  E4M3Sat,
  E5M2Inf,
  E5M2Sat,
  End,
};

class DynamicQuantMatMulFp8 final : public OpKernel {
 public:
  DynamicQuantMatMulFp8(const OpKernelInfo& info);

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_B_SCALE = 2,
    IN_B_ZERO_POINT = 3,
    IN_Y_SCALE = 4,
    IN_Y_ZERO_POINT = 5
  };

  enum OutputTensors : int { OUT_Y = 0 };

  static Status GetFp8Type(const Tensor& tensor, Fp8Mode& out_type);
  static Status GetFp8Type(ONNX_NAMESPACE::TensorProto_DataType elem_type, Fp8Mode& out_type);

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   gsl::span<const size_t> prepacked_buffer_sizes,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  static constexpr int GetBIdx() { return IN_B; }
  IAllocatorUniquePtr<void> quantized_b_;
  size_t quantized_b_size_{0};
  IAllocatorUniquePtr<void> b_scales_;
  size_t b_scale_count_{0};
  bool constant_b_scale_values_validated_{false};
  bool constant_b_zero_point_values_validated_{false};
  TensorShape b_shape_;
  Fp8Mode b_type_{Fp8Mode::E4M3Inf};
  bool has_b_type_{false};
  Fp8Mode fp8_type_{Fp8Mode::E4M3Inf};
  size_t block_size_k_{128};
  size_t block_size_n_{128};
};

}  // namespace contrib
}  // namespace onnxruntime
