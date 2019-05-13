// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif
#include "unique.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(Unique,
                        kMSDomain,
                        1,
                        kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        Unique<float>);

template <>
Status Unique<float>::Compute(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);

  // validate input
  if (input->Shape().NumDimensions() != 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor to Unique op should be 1D");

  // obtain raw input data
  const float* input_data = input->Data<float>();
  size_t num_elements = static_cast<size_t>(input->Shape().Size());

  // 'idx' output has same output shape as input
  Tensor* output_idx = ctx->Output(1, input->Shape());
  int64_t* output_idx_data = output_idx->template MutableData<int64_t>();

  // container to hold the unique elements (in the order it was first seen)
  std::vector<float> unique_elements;
  // number of unique elements is atmost number of elements in the raw data
  unique_elements.reserve(num_elements);

  // containers to store other metadata needed for other output tensors
  std::unordered_map<float, size_t> mapped_indices;
  std::unordered_map<float, size_t> element_counts;

  // processing
  for (size_t i = 0; i < num_elements; ++i) {
    float temp = input_data[i];

    const auto iter = mapped_indices.find(temp);
    if (iter == mapped_indices.end()) {
      // element is being seen for the first time
      element_counts[temp] = 1;
      output_idx_data[i] = mapped_indices[temp] = unique_elements.size();
      unique_elements.push_back(temp);
    } else {
      // element has been seen before
      output_idx_data[i] = iter->second;
      ++element_counts[temp];
    }
  }

  // 'uniques' output
  TensorShape output_shape({static_cast<int64_t>(unique_elements.size())});
  Tensor* output_uniques = ctx->Output(0, output_shape);
  float* output_uniques_data = output_uniques->template MutableData<float>();

  // 'counts' output
  Tensor* output_counts = ctx->Output(2, output_shape);
  int64_t* output_counts_data = output_counts->template MutableData<int64_t>();

  size_t iter = 0;
  for (const float& e : unique_elements) {
    // 'uniques' data
    output_uniques_data[iter] = e;

    // 'counts' data
    const auto iter_map = element_counts.find(e);
    output_counts_data[iter] = static_cast<int64_t>(iter_map->second);

    ++iter;
  }

  return Status::OK();
}

}  // namespace contrib
};  // namespace onnxruntime
