// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class SequenceLength final : public OpKernel {
 public:
  SequenceLength(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class SequenceAt final : public OpKernel {
 public:
  SequenceAt(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class SequenceEmpty final : public OpKernel {
 public:
  SequenceEmpty(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t dtype_{};
};

class SequenceInsert final : public OpKernel {
 public:
  SequenceInsert(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* context) const override;
};

class SequenceErase final : public OpKernel {
 public:
  SequenceErase(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* context) const override;
};

class SequenceConstruct final : public OpKernel {
 public:
  SequenceConstruct(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* context) const override;
};

class SplitToSequence final : public OpKernel {
 public:
  SplitToSequence(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeImpl(OpKernelContext& context, const Tensor& input, const Tensor* p_split_input) const;
  Status PrepareForCompute(const TensorShape& input_shape, int64_t split_scalar, bool is_split_input_scalar,
                           int64_t& num_outputs, int64_t& axis, int& before_dims,
                           int& after_dims_including_split_axis, int& after_dims_excluding_split,
                           bool& is_uneven_split, int& num_remaining_splits,
                           InlinedVector<int64_t>& split_sizes) const;
  int64_t axis_{};
  int64_t keepdims_{1};
  const int64_t DEFAULT_LENGTH_EACH_OUTPUT_ = 1;
};
}  // namespace onnxruntime
