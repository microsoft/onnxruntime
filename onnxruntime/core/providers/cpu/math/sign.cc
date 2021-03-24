// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Sign, Input, 0, element_type_lists::AllNumeric);
}

using SignDataTypes = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Sign, Input, 0);
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
                                      BuildKernelDefConstraintsFromTypeList<SignDataTypes>(),
                                      BuildKernelDefConstraintsFromTypeList<EnabledSignDataTypes>()),
    Sign);

ONNX_CPU_OPERATOR_KERNEL(
    Sign,
    13,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<SignDataTypes>(),
                                      BuildKernelDefConstraintsFromTypeList<EnabledSignDataTypes>()),
    Sign);

namespace sign_internal {
template <class T>
struct CallSignImpl {
  void operator()(const Tensor* input, Tensor* output) const {
    EigenMap<T>(*output) = EigenMap<T>(*input).array().cwiseSign();
  }
};

// The spec does not specify how NaN is
// treated but we have to treat it somehow. We choose
// to return 0 for NaN as TF does.
template <class T>
inline T FloatingImpl(T val) {
  if (std::isnan(val) || val == T(0)) {
    return T(0);
  }
  if (val > T(0)) {
    return T(1);
  } else {
    return T(-1);
  }
}

template <>
struct CallSignImpl<MLFloat16> {
  void operator()(const Tensor* input, Tensor* output) const {
    auto span = gsl::make_span(input->Data<MLFloat16>(), input->Shape().Size());
    auto output_data = output->template MutableData<MLFloat16>();
    std::transform(span.cbegin(), span.cend(), output_data, [](const MLFloat16& val) {
      float fl = math::halfToFloat(val.val);
      return MLFloat16(math::floatToHalf(FloatingImpl(fl)));
    });
  }
};

template <>
struct CallSignImpl<BFloat16> {
  void operator()(const Tensor* input, Tensor* output) const {
    auto span = gsl::make_span(input->Data<BFloat16>(), input->Shape().Size());
    auto output_data = output->template MutableData<BFloat16>();
    std::transform(span.cbegin(), span.cend(), output_data, [](const BFloat16& val) {
      float fl = val.ToFloat();
      return BFloat16(FloatingImpl(fl));
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
