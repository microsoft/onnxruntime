// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif
#include "unique.h"
#include "core/common/inlined_containers.h"
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
  const auto num_elements = input->Shape().Size();

  // 'idx' output has same output shape as input
  Tensor* output_idx = ctx->Output(1, input->Shape());
  int64_t* output_idx_data = output_idx->MutableData<int64_t>();

  struct ElementData {
    int64_t input_pos_; // original index
    int64_t output_pos_;
    int64_t count_; // number of times encountered
  };

  // XXX: Refactoring for less memory allocations. unordered_map
  // used originally for float uniqueness, is this correct?
  using IndexingMap = InlinedHashMap<float, ElementData>;
  IndexingMap mapped_indices;
  mapped_indices.reserve(onnxruntime::narrow<size_t>(num_elements));

  // processing
  for (int64_t i = 0; i < num_elements; ++i) {
    float value = input_data[i];

    const auto original_index = i;
    const auto num_unique = static_cast<int64_t>(mapped_indices.size());
    auto insert_result = mapped_indices.emplace(value, ElementData{original_index, num_unique, 1});
    if (insert_result.second) {
      output_idx_data[i] = num_unique;
    } else {
      // Seen before
      output_idx_data[i] = insert_result.first->second.output_pos_;
      insert_result.first->second.count_++;
    }
  }

  // 'uniques' output
  TensorShape output_shape({static_cast<int64_t>(mapped_indices.size())});
  Tensor* output_uniques = ctx->Output(0, output_shape);
  float* output_uniques_data = output_uniques->MutableData<float>();

  // 'counts' output
  Tensor* output_counts = ctx->Output(2, output_shape);
  int64_t* output_counts_data = output_counts->MutableData<int64_t>();

  for (const auto& e : mapped_indices) {
    // 'uniques' data
    const auto output_pos = e.second.output_pos_;
    output_uniques_data[output_pos] = e.first;
    // 'counts' data
    output_counts_data[output_pos] = e.second.count_;
  }

  return Status::OK();
}

}  // namespace contrib
};  // namespace onnxruntime
