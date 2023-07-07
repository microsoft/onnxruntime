// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

constexpr size_t KSplitViewMaxOutputCount = 16;  // limit of output count.

std::vector<std::pair<int, int>> SplitViewAliasMapping() {
  std::vector<std::pair<int, int>> alias_pairs{};
  for (size_t i = 0; i < KSplitViewMaxOutputCount; ++i) {
    alias_pairs.emplace_back(std::make_pair(0, static_cast<int>(i)));
  }
  return alias_pairs;
}

Status PrepareForSplitView(const Tensor& input_tensor, int64_t num_outputs, const Tensor* p_split_tensor,
                           InlinedVector<TensorShape>& output_shapes, InlinedVector<size_t>& output_offsets) {
  std::vector<int64_t> split_sizes;
  TensorShapeVector input_shape = input_tensor.Shape().AsShapeVector();
  if (p_split_tensor) {
    ORT_ENFORCE(p_split_tensor->Shape().NumDimensions() == 1, "An split tensor must be a vector tensor.");
    auto num_dims = static_cast<size_t>(p_split_tensor->Shape()[0]);
    split_sizes.resize(num_dims);
    const int64_t* data = p_split_tensor->Data<int64_t>();
    split_sizes.assign(data, data + num_dims);
  } else {
    const int64_t split_dim_size = input_shape[0];
    int32_t size = narrow<int32_t>(std::ceil(float(split_dim_size) / num_outputs));
    int32_t remainder = split_dim_size % size;
    split_sizes = std::vector<int64_t>(num_outputs, size);
    if (remainder) {
      split_sizes.back() = remainder;
    }
  }

  size_t bytes_per_elem = input_tensor.DataType()->Size();
  size_t byte_offset = 0;
  for (size_t i = 0; i < split_sizes.size(); ++i) {
    input_shape[0] = split_sizes[i];
    output_shapes.emplace_back(TensorShape(input_shape));
    output_offsets.emplace_back(byte_offset);
    byte_offset += output_shapes[i].Size() * bytes_per_elem;
  }

  if (byte_offset != input_tensor.SizeInBytes()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The input view shapes doesn't adds up to match input buffer size.");
  }

  return Status::OK();
}

class SplitView final : public OpKernel {
 public:
  SplitView(const OpKernelInfo& info) : OpKernel(info) {
    num_outputs_ = info.GetAttrOrDefault<int64_t>("num_outputs", -1);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t num_outputs_ = -1;
};

}  // namespace contrib
}  // namespace onnxruntime
