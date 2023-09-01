// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <numeric>

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
  Status PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                           int& after_dims_including_split_axis, int& after_dims_excluding_split,
                           std::vector<int64_t>& split_sizes) const;

 protected:
  SplitBase(const OpKernelInfo& info, uint32_t opset) : opset_{opset} {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);

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
      num_outputs_ = info.GetAttrOrDefault<int64_t>("num_outputs", -1);
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
