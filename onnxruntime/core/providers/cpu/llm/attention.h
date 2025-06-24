// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

enum class QKMatMulOutputMode {
  kNone = 0,  // No output
  kQK = 1,    // Output QK matrix
  kQKV = 2,   // Output QK and V matrices
};

template <typename T>
class Attention final : public OpKernel {
 public:
  Attention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  bool is_causal;
  int kv_num_heads;
  int q_num_heads;
  QKMatMulOutputMode qk_matmul_output_mode;
  float scale;
  float softcap;
  int softmax_precision;
};

}  // namespace onnxruntime