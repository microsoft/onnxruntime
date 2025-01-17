// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <memory>
#include <vector>

namespace onnxruntime {

class TfIdfVectorizer final : public OpKernel {
 public:
  explicit TfIdfVectorizer(const OpKernelInfo& info);
  ~TfIdfVectorizer() override;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TfIdfVectorizer);

  Status Compute(OpKernelContext* ctx) const override;

 private:
  void ComputeImpl(const void* x_data_raw, size_t elem_size, ptrdiff_t row_num, size_t row_size, bool is_input_string,
                   gsl::span<float> output_data, std::function<void(size_t, gsl::span<float>&)>& fn_weight) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace onnxruntime
