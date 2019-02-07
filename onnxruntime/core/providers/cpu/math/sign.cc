// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"

#include "gsl/span"
#include <type_traits>

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {

class Sign final : public OpKernel {
 public:
  explicit Sign(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};

#define ADD_TYPED_SIGN_OP(data_type)                                       \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                          \
      Sign,                                                                \
      9,                                                                   \
      data_type,                                                           \
      KernelDefBuilder()                                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<data_type>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<data_type>()), \
      Sign);

ADD_TYPED_SIGN_OP(MLFloat16);
ADD_TYPED_SIGN_OP(BFloat16);
ADD_TYPED_SIGN_OP(float);
ADD_TYPED_SIGN_OP(double);

ADD_TYPED_SIGN_OP(int8_t);
ADD_TYPED_SIGN_OP(int16_t);
ADD_TYPED_SIGN_OP(int32_t);
ADD_TYPED_SIGN_OP(int64_t);

ADD_TYPED_SIGN_OP(uint8_t);
ADD_TYPED_SIGN_OP(uint16_t);
ADD_TYPED_SIGN_OP(uint32_t);
ADD_TYPED_SIGN_OP(uint64_t);

namespace sign_internal {
// Unsigned types can only be eq or gt zero
// Signed can be lt, gt or eq to zero
// float, float16 and double will require special handling bc
// - float16 requires unpacking
// - all of then require care for comparing to zeros

// Unsigned Integer types
template <class T>
static void SignUnsigned(const Tensor* input, Tensor* output) {
  static_assert(std::numeric_limits<T>::is_integer &&
                    !std::numeric_limits<T>::is_signed,
                "Expect a unsigned integer type");
  auto span = gsl::make_span(input->Data<T>(), input->Shape().Size());
  auto output_data = output->template MutableData<T>();
  std::transform(span.cbegin(), span.cend(), output_data, [](T val) {
    return (val == T(0)) ? T(0) : T(1);
  });
}

// Signed types
template <class T>
void SignSignedInteger(const Tensor* input, Tensor* output) {
  static_assert(std::numeric_limits<T>::is_integer &&
                    std::numeric_limits<T>::is_signed,
                "Expect a signed type");
  auto span = gsl::make_span(input->Data<T>(), input->Shape().Size());
  auto output_data = output->template MutableData<T>();
  std::transform(span.cbegin(), span.cend(), output_data, [](T val) {
    if (val > T(0)) {
      return T(1);
    } else if (val < T(0)) {
      return T(-1);
    }
    return T(0);
  });
}

// The spec does not specify how NaN is
// treated but we have to treat it somehow. We choose
// to return 0 for NaN as TF does.
template <class T>
inline T FloatingImpl(T val) {
  if (std::isnan(val) || val == T(0)) {
    return T(0);
  } else if (val > T(0)) {
    return T(1);
  } else {
    return T(-1);
  }
}

template <class T>
void SignFloat(const Tensor* input, Tensor* output) {
  static_assert((std::is_same<T, float>::value || std::is_same<T, double>::value),
                "Expect a signed type");
  auto span = gsl::make_span(input->Data<T>(), input->Shape().Size());
  auto output_data = output->template MutableData<T>();
  std::transform(span.cbegin(), span.cend(), output_data, [](T val) {
    return FloatingImpl<T>(val);
  });
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

}  // namespace sign_internal

Status Sign::Compute(OpKernelContext* ctx) const {
  using namespace sign_internal;

  auto input = ctx->Input<Tensor>(0);
  auto output = ctx->Output(0, input->Shape());

  auto dtype = input->DataType();
  if (dtype == DataTypeImpl::GetType<int8_t>()) {
    SignSignedInteger<int8_t>(input, output);
  } else if (dtype == DataTypeImpl::GetType<int16_t>()) {
    SignSignedInteger<int16_t>(input, output);
  } else if (dtype == DataTypeImpl::GetType<int32_t>()) {
    SignSignedInteger<int32_t>(input, output);
  } else if (dtype == DataTypeImpl::GetType<int64_t>()) {
    SignSignedInteger<int64_t>(input, output);
  } else if (dtype == DataTypeImpl::GetType<uint8_t>()) {
    SignUnsigned<uint8_t>(input, output);
  } else if (dtype == DataTypeImpl::GetType<uint16_t>()) {
    SignUnsigned<uint16_t>(input, output);
  } else if (dtype == DataTypeImpl::GetType<uint32_t>()) {
    SignUnsigned<uint32_t>(input, output);
  } else if (dtype == DataTypeImpl::GetType<uint64_t>()) {
    SignUnsigned<uint64_t>(input, output);
  } else if (dtype == DataTypeImpl::GetType<float>()) {
    SignFloat<float>(input, output);
  } else if (dtype == DataTypeImpl::GetType<double>()) {
    SignFloat<double>(input, output);
  } else if (dtype == DataTypeImpl::GetType<MLFloat16>()) {
    SignMLFloat16(input, output);
  } else if (dtype == DataTypeImpl::GetType<BFloat16>()) {
    SignBFloat16(input, output);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported input datatype");
  }
  return Status::OK();
}

}  // namespace onnxruntime
