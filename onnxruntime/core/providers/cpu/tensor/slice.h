// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <iterator>

#include "core/common/narrow.h"

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#endif

#include "core/providers/cpu/tensor/slice_compute_metadata.h"
#include "core/providers/cpu/tensor/slice_helper.h"

namespace onnxruntime {

namespace slice_detail {
template <typename T>
inline void CopyInputData(const Tensor& start_tensor,
                          const Tensor& ends_tensor,
                          const Tensor* axes_tensor,
                          const Tensor* steps_tensor,
                          TensorShapeVector& input_starts,
                          TensorShapeVector& input_ends,
                          TensorShapeVector& input_axes,
                          TensorShapeVector& input_steps) {
  auto start_data = start_tensor.DataAsSpan<T>();
  std::copy(start_data.begin(), start_data.end(), std::back_inserter(input_starts));
  auto ends_data = ends_tensor.DataAsSpan<T>();
  std::copy(ends_data.begin(), ends_data.end(), std::back_inserter(input_ends));
  if (nullptr != axes_tensor) {
    auto axes_data = axes_tensor->DataAsSpan<T>();
    std::copy(axes_data.begin(), axes_data.end(), std::back_inserter(input_axes));
  }
  if (nullptr != steps_tensor) {
    auto steps_data = steps_tensor->DataAsSpan<T>();
    std::copy(steps_data.begin(), steps_data.end(), std::back_inserter(input_steps));
  }
}
}  // namespace slice_detail

class SliceBase {
  // static methods that can be used from other ops if needed
 public:
#ifdef SHARED_PROVIDER
  static Status FlattenOutputDims(gsl::span<const int64_t> input_dimensions, gsl::span<const int64_t> output_dims,
                                  TensorShapeVector& starts, TensorShapeVector& ends, TensorShapeVector& steps,
                                  TensorShapeVector*& p_flattened_input_dims, TensorShapeVector*& p_flattened_output_dims);

  static Status PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                  gsl::span<const int64_t> raw_ends,
                                  gsl::span<const int64_t> raw_axes,
                                  SliceOp::PrepareForComputeMetadata& compute_metadata);

  static Status PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                  gsl::span<const int64_t> raw_ends,
                                  gsl::span<const int64_t> raw_axes,
                                  gsl::span<const int64_t> raw_steps,
                                  SliceOp::PrepareForComputeMetadata& compute_metadata);

  static Status FillVectorsFromInput(const Tensor& start_tensor,
                                     const Tensor& ends_tensor,
                                     const Tensor* axes_tensor,
                                     const Tensor* steps_tensor,
                                     TensorShapeVector& input_starts,
                                     TensorShapeVector& input_ends,
                                     TensorShapeVector& input_axes,
                                     TensorShapeVector& input_steps);
#else
  static inline Status FlattenOutputDims(gsl::span<const int64_t> input_dimensions, gsl::span<const int64_t> output_dims,
                                         TensorShapeVector& starts, TensorShapeVector& ends, TensorShapeVector& steps,
                                         TensorShapeVector*& p_flattened_input_dims, TensorShapeVector*& p_flattened_output_dims) {
    size_t cur = 0;
    size_t nxt = 0;
    while (true) {
      while (nxt < starts.size() && (steps[nxt] != 1 || input_dimensions[nxt] != output_dims[nxt])) {
        p_flattened_input_dims->emplace_back(input_dimensions[nxt]);
        p_flattened_output_dims->emplace_back(output_dims[nxt]);
        starts[cur] = starts[nxt];
        ends[cur] = ends[nxt];
        steps[cur] = steps[nxt];
        ++cur;
        ++nxt;
      }
      if (nxt == starts.size()) {
        break;
      }
      int64_t running_size = 1;
      while (nxt < starts.size() && steps[nxt] == 1 && input_dimensions[nxt] == output_dims[nxt]) {
        running_size *= input_dimensions[nxt];
        ++nxt;
      }
      if (running_size > 1) {
        p_flattened_input_dims->emplace_back(running_size);
        p_flattened_output_dims->emplace_back(running_size);
        starts[cur] = 0LL;
        ends[cur] = running_size;
        steps[cur] = 1LL;
        ++cur;
      }
    }

    if (cur == 0) {
      p_flattened_input_dims->emplace_back(1LL);
      p_flattened_output_dims->emplace_back(1LL);
      starts[cur] = 0LL;
      ends[cur] = 1LL;
      steps[cur] = 1LL;
      ++cur;
    }

    if (p_flattened_output_dims->size() == output_dims.size()) {
      p_flattened_input_dims->clear();
      p_flattened_output_dims->clear();
      p_flattened_input_dims = nullptr;
      p_flattened_output_dims = nullptr;
    } else {
      starts.resize(cur);
      ends.resize(cur);
      steps.resize(cur);
    }

    return Status::OK();
  }

  // compute output_dims without steps (Slice V1-9 & DynamicSlice)
  static inline Status PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                         gsl::span<const int64_t> raw_ends,
                                         gsl::span<const int64_t> raw_axes,
                                         SliceOp::PrepareForComputeMetadata& compute_metadata) {
    ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, compute_metadata));
    ORT_RETURN_IF_ERROR(FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                                          compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_input_dims_,
                                          compute_metadata.p_flattened_output_dims_));
    return Status::OK();
  }

  // compute output_dims with steps (Slice V10)
  static inline Status PrepareForCompute(gsl::span<const int64_t> raw_starts,
                                         gsl::span<const int64_t> raw_ends,
                                         gsl::span<const int64_t> raw_axes,
                                         gsl::span<const int64_t> raw_steps,
                                         SliceOp::PrepareForComputeMetadata& compute_metadata) {
    ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));
    ORT_RETURN_IF_ERROR(FlattenOutputDims(compute_metadata.input_dimensions_, compute_metadata.output_dims_, compute_metadata.starts_,
                                          compute_metadata.ends_, compute_metadata.steps_, compute_metadata.p_flattened_input_dims_,
                                          compute_metadata.p_flattened_output_dims_));
    return Status::OK();
  }

  // Slice V10 & DynamicSlice
  static inline Status FillVectorsFromInput(const Tensor& start_tensor,
                                            const Tensor& ends_tensor,
                                            const Tensor* axes_tensor,
                                            const Tensor* steps_tensor,
                                            TensorShapeVector& input_starts,
                                            TensorShapeVector& input_ends,
                                            TensorShapeVector& input_axes,
                                            TensorShapeVector& input_steps) {
    ORT_RETURN_IF_NOT(start_tensor.Shape().NumDimensions() == 1, "Starts must be a 1-D array");
    ORT_RETURN_IF_NOT(ends_tensor.Shape().NumDimensions() == 1, "Ends must be a 1-D array");
    ORT_RETURN_IF_NOT(start_tensor.Shape() == ends_tensor.Shape(), "Starts and ends shape mismatch");
    ORT_RETURN_IF_NOT(nullptr == axes_tensor || start_tensor.Shape() == axes_tensor->Shape(),
                      "Starts and axes shape mismatch");
    ORT_RETURN_IF_NOT(nullptr == steps_tensor || start_tensor.Shape() == steps_tensor->Shape(),
                      "Starts and steps shape mismatch");

    const auto size = onnxruntime::narrow<size_t>(start_tensor.Shape().Size());
    input_starts.reserve(size);
    input_ends.reserve(size);
    if (nullptr != axes_tensor)
      input_axes.reserve(size);
    if (nullptr != steps_tensor)
      input_steps.reserve(size);

    if (start_tensor.IsDataType<int32_t>()) {
      slice_detail::CopyInputData<int32_t>(start_tensor, ends_tensor, axes_tensor, steps_tensor, input_starts, input_ends, input_axes, input_steps);
    } else if (start_tensor.IsDataType<int64_t>()) {
      slice_detail::CopyInputData<int64_t>(start_tensor, ends_tensor, axes_tensor, steps_tensor, input_starts, input_ends, input_axes, input_steps);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Data type for starts and ends inputs' is not supported in this build. Got ",
                             start_tensor.DataType());
    }

    return Status::OK();
  }
#endif  // SHARED_PROVIDER

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

  // Tag-dispatched constructor for CUDA provider / CUDA plugin builds.
  struct CudaProviderTag {};

  template <typename KernelInfoType>
  SliceBase(const KernelInfoType& info, bool dynamic, CudaProviderTag)
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
