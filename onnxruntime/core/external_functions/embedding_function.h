// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/dlpack/dlpack_converter.h"

namespace onnxruntime {
namespace external_functions {

typedef DLManagedTensor* (*ATenEmbedding)(const DLManagedTensor* weight, const DLManagedTensor* indices,
                                          int64_t padding_idx, bool scale_grad_by_freq);
typedef DLManagedTensor* (*ATenEmbeddingBackward)(const DLManagedTensor* grad, const DLManagedTensor* weight,
                                                  const DLManagedTensor* indices, int64_t padding_idx,
                                                  bool scale_grad_by_freq);

class ATenEmbeddingFunction : public OpKernel {
 public:
  ATenEmbeddingFunction(const OpKernelInfo& info, void* p_fn_raw);
  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  int64_t padding_idx_ = -1;
  bool scale_grad_by_freq_ = false;

  ATenEmbedding p_fn_;
};

class ATenEmbeddingBackwardFunction : public OpKernel {
 public:
  ATenEmbeddingBackwardFunction(const OpKernelInfo& info, void* p_fn_raw);
  Status Compute(OpKernelContext* p_ctx) const override;

 private:
  int64_t padding_idx_ = -1;
  bool scale_grad_by_freq_ = false;

  ATenEmbeddingBackward p_fn_;
};

}  // namespace external_functions
}  // namespace onnxruntime
