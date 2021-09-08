// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/common/type_list.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_utils.h"
#include "core/framework/tensor.h"
#include <cmath>

namespace onnxruntime {
namespace contrib {

class Binarizer final : public OpKernel {
 public:
  explicit Binarizer(const OpKernelInfo& info)
      : OpKernel(info), threshold_(info.GetAttrOrDefault<float>("threshold", 1.0f)) {
  }

  Status Compute(OpKernelContext*) const override;

 private:
  float threshold_;
};

namespace {
using SupportedTypesTypeList = TypeList<float, double, int64_t, int32_t>;
}

// XXX: When/if we merge, this should go to ml domain
ONNX_OPERATOR_KERNEL_EX(
    Binarizer,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefSparseConstraintsFromTypeList<SupportedTypesTypeList>()),
    Binarizer);

namespace {
// XXX: This can be re-used for Dense ml.Binarizer implementation
template<typename T>
struct ThresholdFilter {
  Status operator()(const Tensor& input_values, float threshold, Tensor& output_values) const {
    auto values_span = input_values.DataAsSpan<T>();
    auto* output_data = output_values.MutableData<T>();
    for (auto v : values_span) {
      *output_data++ = (v > threshold) ? static_cast<T>(1) : static_cast<T>(0);
    }
    return Status::OK();
  }
};

template <>
struct ThresholdFilter<double> {
  Status operator()(const Tensor& input_values, double threshold, Tensor& output_values) const {
    auto values_span = input_values.DataAsSpan<double>();
    auto* output_data = output_values.MutableData<double>();
    for (auto v : values_span) {
      if (std::isnan(v)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input data is NaN");
      }
      *output_data++ = (v > threshold) ? 1.0 : 0.0;
    }
    return Status::OK();
  }
};

template <>
struct ThresholdFilter<float> {
  Status operator()(const Tensor& input_values, float threshold, Tensor& output_values) const {
    auto values_span = input_values.DataAsSpan<float>();
    auto* output_data = output_values.MutableData<float>();
    for (auto v : values_span) {
      if (std::isnan(v)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input data is NaN");
      }
      *output_data++ = (v > threshold) ? 1.0f : 0.0f;
    }
    return Status::OK();
  }
};

}

Status Binarizer::Compute(OpKernelContext* ctx) const {
  const SparseTensor& input_tensor = *ctx->Input<SparseTensor>(0);
  ORT_RETURN_IF_NOT(input_tensor.Format() == SparseFormat::kCoo, "Support only COO format currently");

  const TensorShape& input_shape = input_tensor.DenseShape();
  const auto input_indices_span = input_tensor.AsCoo().Indices().DataAsSpan<int64_t>();
  SparseTensor& output_tensor = *ctx->OutputSparse(0, input_shape);
  auto output_mutator = output_tensor.MakeCooData(input_tensor.NumValues(), input_indices_span.size());

  utils::MLTypeCallDispatcherFromTypeList<SupportedTypesTypeList> t_disp(input_tensor.GetElementType());
  auto status = t_disp.InvokeRet<Status, ThresholdFilter>(input_tensor.Values(), threshold_, output_mutator.Values());
  if (!status.IsOK()) return status;
  memcpy(output_mutator.Indices().MutableDataRaw(), input_indices_span.data(), input_indices_span.size_bytes());

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  // DISABLE_SPARSE_TENSORS
