// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/op_kernel.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_utils.h"
#include "core/framework/tensor.h"

#include <unordered_map>

namespace onnxruntime {
namespace contrib {
class OneHotEncoder final : public OpKernel {
 public:
  explicit OneHotEncoder(const OpKernelInfo& info)
      : OpKernel(info),
        zeros_(info.GetAttrOrDefault<int64_t>("zeros", 1) != 0),
        cats_int64s_(),
        num_categories_(0) {
    std::vector<int64_t> cats_int64s = info.GetAttrsOrDefault<int64_t>("cats_int64s");
    ORT_ENFORCE(!cats_int64s.empty(), "'cats_int64' attribute must be defined");
    int64_t idx = 0;
    for (auto v : cats_int64s) {
      cats_int64s_.emplace(v, idx++);
    }
    num_categories_ = gsl::narrow<int64_t>(cats_int64s_.size());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool zeros_;
  std::unordered_map<int64_t, int64_t> cats_int64s_;
  int64_t num_categories_;
};

ONNX_OPERATOR_KERNEL_EX(
    OneHotEncoder,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<int64_t>())
        .TypeConstraint("T1", BuildKernelDefSparseConstraints<float>()),
    OneHotEncoder);

Status OneHotEncoder::Compute(OpKernelContext* ctx) const {
  const auto& input_tensor = *ctx->Input<Tensor>(0);
  const TensorShape& input_shape = input_tensor.Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 0 || input_shape.NumDimensions() == 1,
                    "Expecting a Scalar or 1-D input");

  std::vector<int64_t> output_dims(input_shape.GetDims());
  output_dims.push_back(num_categories_);
  TensorShape output_shape(std::move(output_dims));

  SparseTensor& output_tensor = *ctx->OutputSparse(0, output_shape);
  auto input_span = input_tensor.DataAsSpan<int64_t>();
  std::vector<int64_t> collected_indices;
  auto const cat_end = cats_int64s_.cend();

  for (size_t i = 0, input_size = input_span.size(); i < input_size; ++i) {
    auto v = input_span[i];
    auto hit = cats_int64s_.find(v);
    ORT_RETURN_IF(!zeros_ && hit == cat_end, "Input element: ", v, " at [", i, "] could not categorized");
    auto index = i * num_categories_ + hit->second;
    collected_indices.push_back(index);
  }

  if (collected_indices.empty()) {
    ORT_UNUSED_PARAMETER(output_tensor.MakeCooData(0, 0));
  } else {
    const auto sparse_size = collected_indices.size();
    auto coo_mutator = output_tensor.MakeCooData(sparse_size, sparse_size);
    std::fill_n(coo_mutator.Values().MutableData<float>(), sparse_size, 1.0f);
    memcpy(coo_mutator.Indices().MutableDataRaw(), collected_indices.data(), sparse_size * sizeof(int64_t));
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  // DISABLE_SPARSE_TENSORS
