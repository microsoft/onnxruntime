// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "core/common/gsl.h"

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/math.h"
#include "core/framework/op_kernel.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math.h"

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Sign, Input, 0, element_type_lists::AllNumeric);
}

using EnabledSignDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Sign, Input, 0);

class Sign final : public OpKernel {
 public:
  explicit Sign(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Sign,
    9,
    12,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSignDataTypes>()),
    Sign);

ONNX_CPU_OPERATOR_KERNEL(
    Sign,
    13,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledSignDataTypes>()),
    Sign);

namespace sign_internal {
template <class T>
struct CallSignImpl {
  void operator()(const Tensor* input, Tensor* output) const {
    EigenMap<T>(*output) = EigenMap<T>(*input).array().cwiseSign();
  }
};

template <>
struct CallSignImpl<MLFloat16> {
  void operator()(const Tensor* input, Tensor* output) const {
    auto span = input->DataAsSpan<MLFloat16>();
    auto output_data = output->MutableData<MLFloat16>();
    std::transform(span.begin(), span.end(), output_data, [](const MLFloat16& val) {
      // Return 0 as TF does for NaN.
      if (val.IsNaNOrZero()) return MLFloat16::Zero;
      return (val.IsNegative()) ? MLFloat16::MinusOne : MLFloat16::One;
    });
  }
};

template <>
struct CallSignImpl<BFloat16> {
  void operator()(const Tensor* input, Tensor* output) const {
    auto span = input->DataAsSpan<BFloat16>();
    auto output_data = output->MutableData<BFloat16>();
    std::transform(span.begin(), span.end(), output_data, [](const BFloat16& val) {
      // Return 0 as TF does for NaN.
      if (val.IsNaNOrZero()) return BFloat16::Zero;
      return (val.IsNegative()) ? BFloat16::MinusOne : BFloat16::One;
    });
  }
};
}  // namespace sign_internal

Status Sign::Compute(OpKernelContext* ctx) const {
  using namespace sign_internal;

  auto input = ctx->Input<Tensor>(0);
  auto output = ctx->Output(0, input->Shape());

  const auto dtype = input->GetElementType();
  utils::MLTypeCallDispatcherFromTypeList<EnabledSignDataTypes> t_disp(dtype);
  t_disp.Invoke<CallSignImpl>(input, output);

  return Status::OK();
}

}  // namespace onnxruntime
