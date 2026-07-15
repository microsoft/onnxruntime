// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <numeric>

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

class SplitBase {
 public:
  /*
   * \param num_outputs must >=0
   */
#ifdef SHARED_PROVIDER
  Status PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                           int& after_dims_including_split_axis, int& after_dims_excluding_split,
                           std::vector<int64_t>& split_sizes) const;
#else
  inline Status PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                                  int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                  std::vector<int64_t>& split_sizes) const {
    auto input_dims = input_shape.GetDims();
    const auto num_dimensions = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
    axis = HandleNegativeAxis(axis_, num_dimensions);
    const int64_t split_dim_size = input_dims[onnxruntime::narrow<size_t>(axis)];

    before_dims = onnxruntime::narrow<int>(input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis)));
    after_dims_including_split_axis = onnxruntime::narrow<int>(input_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis)));
    after_dims_excluding_split = (axis + 1 == num_dimensions)
                                     ? 1
                                     : onnxruntime::narrow<int>(input_shape.SizeFromDimension(SafeInt<size_t>(axis) + 1));

    if (num_outputs_ != -1) {
      if (num_outputs_ > split_dim_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid num_outputs value of ", num_outputs_,
                               ". Size of dimension being split is ", split_dim_size);
      }
      int32_t size = onnxruntime::narrow<int32_t>(std::ceil(static_cast<float>(split_dim_size) / num_outputs));
      int32_t remainder = split_dim_size % size;
      split_sizes = std::vector<int64_t>(num_outputs, size);
      if (remainder) {
        split_sizes.back() = remainder;
      }
    }

    if (split_sizes.empty()) {
      if (split_dim_size % static_cast<size_t>(num_outputs) != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input cannot be split evenly on selected axis. Input shape=", input_shape,
                               " Axis=", axis_, " NumOutputs=", num_outputs);
      }
      split_sizes = std::vector<int64_t>(static_cast<size_t>(num_outputs), split_dim_size / num_outputs);
    } else {
      int64_t split_size_sum = split_size_sum_;
      if (split_size_sum == -1) {
        split_size_sum = std::accumulate(split_sizes.cbegin(), split_sizes.cend(), 0LL);
      }
      if (split_sizes.size() != static_cast<size_t>(num_outputs) || split_size_sum != split_dim_size)
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Cannot split using values in 'split' attribute. Axis=", axis_,
                               " Input shape=", input_shape,
                               " NumOutputs=", num_outputs,
                               " Num entries in 'split' (must equal number of outputs) was ", split_sizes.size(),
                               " Sum of sizes in 'split' (must equal size of selected axis) was ", split_size_sum);
    }
    return Status::OK();
  }
#endif  // SHARED_PROVIDER

 protected:
  template <typename KernelInfoType>
  SplitBase(const KernelInfoType& info, uint32_t opset) : opset_{opset} {
    axis_ = info.template GetAttrOrDefault<int64_t>("axis", 0);

    size_t num_inputs = info.GetInputCount();
    if (num_inputs == 1) {
      // optional
      if (info.GetAttrs("split", split_sizes_).IsOK()) {
        split_size_sum_ = std::accumulate(split_sizes_.cbegin(), split_sizes_.cend(), 0LL);
        ORT_ENFORCE(std::all_of(split_sizes_.cbegin(), split_sizes_.cend(), [](int64_t value) { return value >= 0; }),
                    "Invalid value in 'split' attribute. All values must be > 0");
      }
    }

    if (opset_ >= 18) {
      num_outputs_ = info.template GetAttrOrDefault<int64_t>("num_outputs", -1);
      // the ONNX type/shape inferencing handles the check that num_outputs is > 0
      // ORT_ENFORCE(num_outputs_ != 0, "Invalid value in 'num_outputs' attribute of 0.");

      if (num_outputs_ != -1 && info.GetInputCount() == 2) {
        ORT_THROW("If 'num_outputs' is specified, the 'split' input should not be provided.");
      }
    }
  }

  const uint32_t opset_;
  int64_t axis_;
  std::vector<int64_t> split_sizes_;
  int64_t split_size_sum_ = -1;
  int64_t num_outputs_ = -1;
};

class SplitImpl : public OpKernel, public SplitBase {
 public:
  SplitImpl(const OpKernelInfo& info, uint32_t opset) : OpKernel(info), SplitBase(info, opset) {}

  Status Compute(OpKernelContext* context) const override;
};

// versions 1, 2, 11 and 13
class Split_1_13 final : public SplitImpl {
 public:
  Split_1_13(const OpKernelInfo& info) : SplitImpl(info, 1) {}
};

class Split_18 final : public SplitImpl {
 public:
  Split_18(const OpKernelInfo& info) : SplitImpl(info, 18) {}
};

}  // namespace onnxruntime
