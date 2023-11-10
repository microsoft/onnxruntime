// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>

#include "core/common/gsl.h"

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/type_list.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/data_types.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math_cpuonly.h"

#include "Eigen/src/Core/arch/Default/BFloat16.h"
#include "Eigen/src/Core/arch/Default/Half.h"

#if defined(_M_AMD64) && !defined(_M_ARM64EC)
#include "core/mlas/inc/mlas.h"
#endif

namespace onnxruntime {

namespace op_kernel_type_control {
// we're using one set of types for all opsets of Cast
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Input, 0,
    element_type_lists::AllIRv9);

ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Input, 0,
    bool, int32_t, int64_t);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Output, 0,
    element_type_lists::AllIRv9);

ORT_SPECIFY_OP_KERNEL_ARG_REQUIRED_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Cast, Output, 0,
    bool, int32_t, int64_t);
}  // namespace op_kernel_type_control

namespace castop_internal {
using EnabledSrcTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                       Cast, Input, 0);
using EnabledDstTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(kCpuExecutionProvider, kOnnxDomain,
                                                                       Cast, Output, 0);

template <typename T>
using IsOrtFloat16Type = boost::mp11::mp_contains<TypeList<BFloat16, MLFloat16>, T>;

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename T>
using IsOrtFloat8Type = boost::mp11::mp_contains<element_type_lists::AllFloat8, T>;
#endif

// string cast helpers
// Note: when C++17 is available, use <charconv> functions

// handle floating point output separately
template <typename SrcType>
typename std::enable_if<std::is_floating_point<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  static_assert(sizeof(SrcType) <= sizeof(double),
                "largest supported floating point type is double");
  if (std::isnan(input)) {
    output = "NaN";
  } else if (std::isinf(input)) {
    if (input < std::numeric_limits<SrcType>::lowest()) {
      output = "-INF";
    } else {
      output = "INF";
    }
  } else {
    // set precision to 8 to match numpy default behavior
    constexpr const char* format = "%.8g";
    const double value = static_cast<double>(input);

    char static_buffer[256];
    std::unique_ptr<char[]> dynamic_buffer{};

    gsl::span<char> buffer_span = gsl::make_span(static_buffer);

    auto snprintf_result = std::snprintf(buffer_span.data(), buffer_span.size(), format, value);
    ORT_ENFORCE(snprintf_result > 0, "snprintf() failed with return value: ", snprintf_result);

    // include trailing '\0'
    const size_t required_buffer_size = gsl::narrow_cast<size_t>(snprintf_result) + 1;

    if (required_buffer_size > buffer_span.size()) {
      // didn't get it all, allocate a bigger buffer and retry
      dynamic_buffer = std::make_unique<char[]>(required_buffer_size);
      buffer_span = gsl::make_span(dynamic_buffer.get(), required_buffer_size);
      snprintf_result = std::snprintf(buffer_span.data(), buffer_span.size(), format, value);
      ORT_ENFORCE(
          snprintf_result > 0 &&
              gsl::narrow_cast<size_t>(snprintf_result) == buffer_span.size() - 1,
          "Failed to write value with snprintf().");
    }

    output.assign(buffer_span.data(), required_buffer_size - 1);
  }
}

template <typename SrcType>
typename std::enable_if<std::is_integral<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  output = std::to_string(input);
}

template <typename SrcType>
#if !defined(DISABLE_FLOAT8_TYPES)
typename std::enable_if<IsOrtFloat16Type<SrcType>::value || IsOrtFloat8Type<SrcType>::value, void>::type
#else
typename std::enable_if<IsOrtFloat16Type<SrcType>::value>::type
#endif
CastToString(const SrcType& input, std::string& output) {
  CastToString(static_cast<float>(input), output);
}

template <typename DstType>
typename std::enable_if<std::is_floating_point<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(double),
                "largest supported floating point type is double");
  output = gsl::narrow_cast<DstType>(std::stod(input));
}

template <typename DstType>
typename std::enable_if<std::is_integral<DstType>::value && std::is_unsigned<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(unsigned long long),
                "largest supported unsigned integral type is unsigned long long");
  output = gsl::narrow_cast<DstType>(std::stoull(input));
}

template <typename DstType>
typename std::enable_if<std::is_integral<DstType>::value && std::is_signed<DstType>::value, void>::type
CastFromString(const std::string& input, DstType& output) {
  static_assert(sizeof(DstType) <= sizeof(long long),
                "largest supported signed integral type is long long");
  output = gsl::narrow_cast<DstType>(std::stoll(input));
}

template <typename DstType>
#if !defined(DISABLE_FLOAT8_TYPES)
typename std::enable_if<IsOrtFloat16Type<DstType>::value || IsOrtFloat8Type<DstType>::value, void>::type
#else
typename std::enable_if<IsOrtFloat16Type<DstType>::value, void>::type
#endif
CastFromString(const std::string& input, DstType& output) {
  float intermediate;
  CastFromString(input, intermediate);
  output = DstType(intermediate);
}

// type that is usable with Eigen cast
template <typename T>
struct EigenCastType {
  using type = T;
};

// ORT float16 types don't support Eigen cast, so map them to Eigen ones

template <>
struct EigenCastType<MLFloat16> {
  using type = Eigen::half;
};

template <>
struct EigenCastType<BFloat16> {
  using type = Eigen::bfloat16;
};
// generic tensor X -> Y
template <typename SrcType, typename DstType, typename Enable = void>
struct TensorCaster {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    using SrcEigenCastType = typename EigenCastType<SrcType>::type;
    using DstEigenCastType = typename EigenCastType<DstType>::type;

    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto in_vector =
        ConstEigenVectorMap<SrcEigenCastType>(reinterpret_cast<const SrcEigenCastType*>(in.Data<SrcType>()), shape_size);
    auto out_vector =
        EigenVectorMap<DstEigenCastType>(reinterpret_cast<DstEigenCastType*>(out.MutableData<DstType>()), shape_size);
    out_vector = in_vector.template cast<DstEigenCastType>();
  }
};

// tensor X -> string
template <typename SrcType>
struct TensorCaster<SrcType, std::string> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<std::string>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastToString(in_data[i], out_data[i]);
    }
  }
};

// tensor string -> X
template <typename DstType>
struct TensorCaster<std::string, DstType> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<DstType>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastFromString(in_data[i], out_data[i]);
    }
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

// tensor X -> float 8
template <typename SrcType, typename DstType, typename Enable = void>
struct TensorCasterNoSat {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<SrcType>();
    auto* out_data = out.MutableData<DstType>();
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      out_data[i] = DstType(static_cast<float>(in_data[i]), false);
    }
  }
};

// tensor string -> float 8
template <typename DstType>
struct TensorCasterNoSat<std::string, DstType> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    const std::ptrdiff_t shape_size = narrow<std::ptrdiff_t>(shape.Size());
    const auto* in_data = in.Data<std::string>();
    auto* out_data = out.MutableData<DstType>();
    float float_value;
    for (std::ptrdiff_t i = 0; i < shape_size; ++i) {
      CastFromString(in_data[i], float_value);
      out_data[i] = DstType(float_value, false);
    }
  }
};

#endif

#if defined(_M_AMD64) && !defined(_M_ARM64EC)
// specializations to use optimized and Windows x64-specific
// MlasConvertHalfToFloatBuffer() routine for MLFloat16 -> float conversion

// tensor MLFloat16 -> float
template <>
struct TensorCaster<MLFloat16, float> {
  void Cast(const OpKernelContext&, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    auto out_data = out.MutableData<float>();
    auto in_data = in.Data<MLFloat16>();
    const size_t shape_size = narrow<size_t>(shape.Size());
    MlasConvertHalfToFloatBuffer(&in_data[0].val, out_data, shape_size);
  }
};

Tensor GetIntermediateMLFloat16ToFloatTensor(
    const OpKernelContext& context, const TensorShape& shape, const Tensor& in) {
  AllocatorPtr allocator;
  ORT_THROW_IF_ERROR(context.GetTempSpaceAllocator(&allocator));
  Tensor out{DataTypeImpl::GetType<float>(), shape, allocator};
  TensorCaster<MLFloat16, float>{}.Cast(context, shape, in, out);
  return out;
}

template <typename DstType>
void CastMLFloat16ThroughFloatTensor(
    const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) {
  // use optimized MLFloat16 -> float, then float -> DstType
  Tensor intermediate_tensor = GetIntermediateMLFloat16ToFloatTensor(context, shape, in);
  TensorCaster<float, DstType>{}.Cast(context, shape, intermediate_tensor, out);
}

// tensor MLFloat16 -> X
template <typename DstType>
struct TensorCaster<MLFloat16, DstType> {
  void Cast(const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    CastMLFloat16ThroughFloatTensor<DstType>(context, shape, in, out);
  }
};

// tensor MLFloat16 -> string
template <>
struct TensorCaster<MLFloat16, std::string> {
  void Cast(const OpKernelContext& context, const TensorShape& shape, const Tensor& in, Tensor& out) const {
    CastMLFloat16ThroughFloatTensor<std::string>(context, shape, in, out);
  }
};
#endif

class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);

    int64_t saturate = info.GetAttrOrDefault("saturate", int64_t{1});
#if !defined(DISABLE_FLOAT8_TYPES)
    if (saturate == 0 && (to != ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FNUZ &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E5M2 &&
                          to != ONNX_NAMESPACE::TensorProto::FLOAT8E5M2FNUZ)) {
      ORT_THROW("Attribute saturate is only used for cast to float 8 types.");
    }
#else
    if (saturate == 0) {
      ORT_THROW("Attribute saturate is only used for cast to float 8 types.");
    }
#endif
    saturate_ = saturate == 1;
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
  bool saturate_;
};

template <typename TSrc, typename TDst>
struct Dispatcher {
  void operator()(const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    TensorCaster<TSrc, TDst>{}.Cast(context, shape, src, dst);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename TSrc, typename TDst>
struct DispatcherNoSat {
  void operator()(const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    TensorCasterNoSat<TSrc, TDst>{}.Cast(context, shape, src, dst);
  }
};

#endif

template <typename TSrc>
struct SrcDispatcher {
  void operator()(
      int32_t to, const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    using EnabledDstTypesWithoutSrcType =
        boost::mp11::mp_remove_if_q<EnabledDstTypes, boost::mp11::mp_bind_front<std::is_same, TSrc>>;
    utils::MLTypeCallDispatcherFromTypeList<EnabledDstTypesWithoutSrcType> dispatcher{to};
    dispatcher.template InvokeWithLeadingTemplateArgs<Dispatcher, TypeList<TSrc>>(context, shape, src, dst);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename TSrc>
struct SrcDispatcherNoSat {
  void operator()(
      int32_t to, const OpKernelContext& context, const TensorShape& shape, const Tensor& src, Tensor& dst) {
    using EnabledDstTypeOnlyFloat8 = boost::mp11::mp_set_intersection<
        EnabledDstTypes, element_type_lists::AllFloat8>;
    using EnabledDstTypesWithoutSrcType =
        boost::mp11::mp_remove_if_q<EnabledDstTypeOnlyFloat8, boost::mp11::mp_bind_front<std::is_same, TSrc>>;
    utils::MLTypeCallDispatcherFromTypeList<EnabledDstTypesWithoutSrcType> dispatcher{to};
    dispatcher.template InvokeWithLeadingTemplateArgs<DispatcherNoSat, TypeList<TSrc>>(context, shape, src, dst);
  }
};

#endif

Status Cast::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);

  if (shape.Size() == 0) {
    return Status::OK();
  }

  const auto from = X->GetElementType();

  if (from == to_) {
    // will copy if X and Y have different buffers
    CopyCpuTensor(X, Y);
    return Status::OK();
  }

#if !defined(DISABLE_FLOAT8_TYPES)
  if (saturate_) {
#endif
    utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
    dispatcher.Invoke<SrcDispatcher>(to_, *context, shape, *X, *Y);
#if !defined(DISABLE_FLOAT8_TYPES)
  } else if (to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FNUZ ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E5M2 ||
             to_ == ONNX_NAMESPACE::TensorProto::FLOAT8E5M2FNUZ) {
    utils::MLTypeCallDispatcherFromTypeList<EnabledSrcTypes> dispatcher{from};
    dispatcher.Invoke<SrcDispatcherNoSat>(to_, *context, shape, *X, *Y);
  }
#endif

  return Status::OK();
}
}  // namespace

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    6,
    12,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<castop_internal::EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<castop_internal::EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    castop_internal::Cast);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    13,
    18,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_KERNEL(
    Cast,
    19,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<castop_internal::EnabledSrcTypes>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<castop_internal::EnabledDstTypes>())
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    castop_internal::Cast);

}  // namespace onnxruntime
