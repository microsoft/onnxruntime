// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#endif

#include "core/providers/cpu/tensor/slice_compute_metadata.h"

namespace onnxruntime {

class SliceBase {
  // static methods that can be used from other ops if needed
 public:
  // compute output_dims without steps (Slice V1-9 & DynamicSlice)
  static Status PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                  gsl::span<const int64_t> raw_ends,
                                  gsl::span<const int64_t> raw_axes,
                                  SliceOp::PrepareForComputeMetadata& compute_metadata);

  // compute output_dims with steps (Slice V10)
  static Status PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                  gsl::span<const int64_t> raw_ends,
                                  gsl::span<const int64_t> raw_axes,
                                  gsl::span<const int64_t> raw_steps,
                                  SliceOp::PrepareForComputeMetadata& compute_metadata);

  // Slice V10 & DynamicSlice
  static Status FillVectorsFromInput(const Tensor& start_tensor,
                                     const Tensor& ends_tensor,
                                     const Tensor* axes_tensor,
                                     const Tensor* steps_tensor,
                                     TensorShapeVector& input_starts,
                                     TensorShapeVector& input_ends,
                                     TensorShapeVector& input_axes,
                                     TensorShapeVector& input_steps);

 protected:
  SliceBase(const OpKernelInfo& info, bool dynamic = false)
      : dynamic_(dynamic) {
    if (!dynamic) {
      auto has_starts = info.GetAttrs("starts", attr_starts_).IsOK();
      auto has_ends = info.GetAttrs("ends", attr_ends_).IsOK();
      auto has_axes = info.GetAttrs("axes", attr_axes_).IsOK();
      ORT_ENFORCE(has_starts && has_ends && attr_starts_.size() == attr_ends_.size(),
                  "Missing or invalid starts and ends attribute");
      ORT_ENFORCE(!has_axes || attr_axes_.size() == attr_starts_.size(),
                  "Invalid axes attribute, axes attribute (if present) should have the same size as starts/ends attributes");
    }
  }

  Status Compute(OpKernelContext* context) const;

 protected:
  gsl::span<const int64_t> StartsAttribute() const { return attr_starts_; }
  gsl::span<const int64_t> EndsAttribute() const { return attr_ends_; }
  gsl::span<const int64_t> AxesAttribute() const { return attr_axes_; }

 private:
  bool dynamic_;
  std::vector<int64_t> attr_starts_, attr_ends_, attr_axes_;
};

struct Slice1 final : public OpKernel, public SliceBase {
  Slice1(const OpKernelInfo& info) : OpKernel(info), SliceBase(info, false) {}
  Status Compute(OpKernelContext* context) const override { return SliceBase::Compute(context); }
};

struct Slice10 final : public OpKernel, public SliceBase {
  Slice10(const OpKernelInfo& info) : OpKernel(info), SliceBase(info, true) {}
  Status Compute(OpKernelContext* context) const override { return SliceBase::Compute(context); }
};

}  // namespace onnxruntime
