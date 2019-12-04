// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "gsl/gsl"
#include <type_traits>

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

class Sign final : public OpKernel {
 public:
  explicit Sign(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};

ONNX_CPU_OPERATOR_KERNEL(
    Sign,
    9,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>(),
                                            DataTypeImpl::GetTensorType<int64_t>(),
                                            DataTypeImpl::GetTensorType<uint64_t>(),
                                            DataTypeImpl::GetTensorType<int32_t>(),
                                            DataTypeImpl::GetTensorType<uint32_t>(),
                                            DataTypeImpl::GetTensorType<int16_t>(),
                                            DataTypeImpl::GetTensorType<uint16_t>(),
                                            DataTypeImpl::GetTensorType<int8_t>(),
                                            DataTypeImpl::GetTensorType<uint8_t>(),
                                            DataTypeImpl::GetTensorType<MLFloat16>(),
                                            DataTypeImpl::GetTensorType<BFloat16>()}),
    Sign);

namespace sign_internal {
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

void SignMLFloat16(const Tensor* input, Tensor* output) {
  auto span = gsl::make_span(input->Data<MLFloat16>(), input->Shape().Size());
  auto output_data = output->template MutableData<MLFloat16>();
  std::transform(span.cbegin(), span.cend(), output_data, [](const MLFloat16& val) {
    float fl = math::halfToFloat(val.val);
    return MLFloat16(math::floatToHalf(FloatingImpl(fl)));
  });
}

void SignBFloat16(const Tensor* input, Tensor* output) {
  auto span = gsl::make_span(input->Data<BFloat16>(), input->Shape().Size());
  auto output_data = output->template MutableData<BFloat16>();
  std::transform(span.cbegin(), span.cend(), output_data, [](const BFloat16& val) {
    float fl = val.ToFloat();
    return BFloat16(FloatingImpl(fl));
  });
}

// MLTypeTypeCallDispatcher functor. This is a generic
// implementation for all types other than MLFloat16 and BFloat16 above.
template <class T>
struct CallSignImpl {
  void operator()(const Tensor* input, Tensor* output) const {
    EigenMap<T>(*output) = EigenMap<T>(*input).array().cwiseSign();
  }
};
}  // namespace sign_internal

Status Sign::Compute(OpKernelContext* ctx) const {
  using namespace sign_internal;

  auto input = ctx->Input<Tensor>(0);
  auto output = ctx->Output(0, input->Shape());

  auto dtype = input->GetElementType();
  switch (dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      SignBFloat16(input, output);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      SignMLFloat16(input, output);
      break;
    default:
      utils::MLTypeCallDispatcher<CallSignImpl, float, double, int8_t, uint8_t,
                                  int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>
          t_disp(dtype);
      t_disp.Invoke(input, output);
      break;
  }
  return Status::OK();
}

}  // namespace onnxruntime
