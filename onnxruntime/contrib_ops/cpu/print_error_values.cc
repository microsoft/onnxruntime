// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef PRINT_ERROR_VALUES

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class PrintErrorValues : public OpKernel {
 public:
  PrintErrorValues(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("node_name", &node_name_).IsOK());
    ORT_ENFORCE(info.GetAttr<std::string>("node_type", &node_type_).IsOK());
    ORT_ENFORCE(info.GetAttr<std::string>("execution_provider", &execution_provider_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("node_output_index", &node_output_index_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);

    auto elements = X->DataAsSpan<T>();
    bool has_nan = false;
    bool has_inf = false;

    for (T element : elements) {
      if (std::isnan(element)) {
        has_nan = true;
      }

      if (std::isinf(element)) {
        has_inf = true;
      }

      if (has_nan && has_inf) {
        break;
      }
    }

    if (has_nan) {
      printf("%s (%s) produced one or more NaN values for output %lld on %s\n",
             node_name_.c_str(),
             node_type_.c_str(),
             node_output_index_,
             execution_provider_.c_str());
    }

    if (has_inf) {
      printf("%s (%s) produced one or more INF values for output %lld on %s\n",
             node_name_.c_str(),
             node_type_.c_str(),
             node_output_index_,
             execution_provider_.c_str());
    }

    Tensor* Y = context->Output(0, X->Shape());
    CopyCpuTensor(X, Y);

    return Status::OK();
  }

 private:
  std::string node_name_;
  std::string node_type_;
  std::string execution_provider_;
  int64_t node_output_index_;
};

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    PrintErrorValues,
    1,
    MLFloat16,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PrintErrorValues<MLFloat16>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    PrintErrorValues,
    1,
    float,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    PrintErrorValues<float>);
}  // namespace contrib
}  // namespace onnxruntime

#endif
