// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iomanip>
#include <sstream>

#include "gsl/gsl"

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "Eigen/src/Core/arch/Default/Half.h"

#if defined(_M_AMD64)
#include "core/mlas/inc/mlas.h"
#endif

#include <boost/mp11.hpp>

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

// test changes
using namespace boost::mp11;
namespace {
template <typename... Ts>
using TypeList = mp_list<Ts...>;

template <typename... Types>
class MLTypeCallDispatcher2 {
  int32_t dt_type_;

 public:
  explicit MLTypeCallDispatcher2(int32_t dt_type) noexcept : dt_type_(dt_type) {}

  template <template <typename> class Fn, typename... Args>
  void Invoke(Args&&... args) const {
    utils::mltype_dispatcher_internal::CallableDispatchableHelper helper(dt_type_);
    int results[] = {0, helper.template Invoke<Types>(Fn<Types>(), std::forward<Args>(args)...)...};
    static_cast<void>(results);
    ORT_ENFORCE(helper.called_ == 1);
  }

  template <template <typename...> class Fn, typename LeadingTemplateArgTypeList, typename... Args>
  void InvokeWithLeadingTemplateArgs(Args&&... args) const {
    utils::mltype_dispatcher_internal::CallableDispatchableHelper helper(dt_type_);
    int results[] = {0, helper.template Invoke<Types>(mp_apply<Fn, mp_push_back<LeadingTemplateArgTypeList, Types>>(), std::forward<Args>(args)...)...};
    static_cast<void>(results);
    ORT_ENFORCE(helper.called_ == 1);
  }
};

//#define LIMIT_TYPES
//#define LIMIT_TYPES_NO_STRING_OR_FLOAT16
#if defined(LIMIT_TYPES)
using ImplementedSrcTypes = TypeList<float, int64_t>;

using ImplementedDstTypes = TypeList<float, int64_t>;
#elif defined(LIMIT_TYPES_NO_STRING_OR_FLOAT16)
using ImplementedSrcTypes = TypeList<
    bool,
    float, double,
    uint8_t, uint16_t, uint32_t, uint64_t,
    int8_t, int16_t, int32_t, int64_t>;

using ImplementedDstTypes = TypeList<
    bool,
    float, double,
    uint8_t, uint16_t, uint32_t, uint64_t,
    int8_t, int16_t, int32_t, int64_t>;
#else
using ImplementedSrcTypes = TypeList<
    bool,
    float, double,
    uint8_t, uint16_t, uint32_t, uint64_t,
    int8_t, int16_t, int32_t, int64_t,
    MLFloat16, BFloat16,
    std::string>;

using ImplementedDstTypes = TypeList<
    bool,
    float, double,
    uint8_t, uint16_t, uint32_t, uint64_t,
    int8_t, int16_t, int32_t, int64_t,
    MLFloat16, BFloat16,
    std::string>;
#endif
}  // namespace

namespace {

using IndirectCastTypes = TypeList<MLFloat16, BFloat16>;

template <typename Type>
using IsDirectCastType = mp_not<mp_contains<IndirectCastTypes, Type>>;

template <typename... Types>
using AreAllDirectCastTypes = mp_all<IsDirectCastType<Types>...>;

// string cast helpers

// handle floating point input separately
template <typename SrcType>
typename std::enable_if<std::is_floating_point<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  if (std::isnan(input)) {
    output = "NaN";
  } else if (std::isinf(input)) {
    if (input < std::numeric_limits<SrcType>::lowest()) {
      output = "-INF";
    } else {
      output = "INF";
    }
  } else {
    // setprecision to 8 to match numpy default behavior
    std::ostringstream convert;
    convert << std::setprecision(8) << input;
    output = convert.str();
  }
}

template <typename SrcType>
typename std::enable_if<!std::is_floating_point<SrcType>::value, void>::type
CastToString(const SrcType& input, std::string& output) {
  std::ostringstream convert;
  convert << input;
  output = convert.str();
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

// generic scalar X -> Y
template <typename SrcType, typename DstType>
struct ScalarDirectCaster {
  void Cast(const SrcType& in, DstType& out) const {
    out = static_cast<DstType>(in);
  }
};

// scalar X -> string
template <typename SrcType>
struct ScalarDirectCaster<SrcType, std::string> {
  void Cast(const SrcType& in, std::string& out) const {
    CastToString<SrcType>(in, out);
  }
};

// scalar string -> X
template <typename DstType>
struct ScalarDirectCaster<std::string, DstType> {
  void Cast(const std::string& in, DstType& out) const {
    CastFromString<DstType>(in, out);
  }
};

// scalar float -> MLFloat16
template <>
struct ScalarDirectCaster<float, MLFloat16> {
  void Cast(const float& in, MLFloat16& out) {
    out = MLFloat16{math::floatToHalf(in)};
  }
};

// helper for indirect cast types
template <typename SrcType, typename DstType, typename IntermediateType>
struct ScalarIndirectCaster {
  void Cast(const SrcType& in, DstType& out) const {
    IntermediateType intermediate;
    ScalarDirectCaster<SrcType, IntermediateType>{}.Cast(in, intermediate);
    ScalarDirectCaster<IntermediateType, DstType>{}.Cast(intermediate, out);
  }
};

template <typename SrcType, typename DstType, class Enable = void>
struct ScalarCaster;

template <typename SrcType, typename DstType>
struct ScalarCaster<
    SrcType, DstType,
    typename std::enable_if<AreAllDirectCastTypes<SrcType, DstType>::value>::type> {
  void Cast(const SrcType& in, DstType& out) const {
    ScalarDirectCaster<SrcType, DstType>{}.Cast(in, out);
  }
};

template <typename SrcType, typename DstType>
struct ScalarCaster<
    SrcType, DstType,
    typename std::enable_if<!AreAllDirectCastTypes<SrcType, DstType>::value>::type> {
  void Cast(const SrcType& in, DstType& out) const {
    ScalarIndirectCaster<SrcType, DstType, float>{}.Cast(in, out);
  }
};

// generic tensor X -> Y
template <typename SrcType, typename DstType>
struct TensorCaster {
  void Cast(const Tensor& in, Tensor& out, const TensorShape& shape) const {
    const auto shape_size = shape.Size();

    // const auto in_vector = ConstEigenVectorMap<SrcType>(in.Data<SrcType>(), shape_size);
    // auto output_vector = EigenVectorMap<DstType>(out.MutableData<DstType>(), shape_size);
    // output_vector = in_vector.template cast<DstType>();

    const auto in_vector = ConstEigenVectorMap<SrcType>(in.Data<SrcType>(), shape_size);
    auto output_vector = EigenVectorMap<DstType>(out.MutableData<DstType>(), shape_size);
    output_vector = in_vector.unaryExpr([](const SrcType& in_scalar) {
      DstType out_scalar;
      ScalarCaster<SrcType, DstType>{}.Cast(in_scalar, out_scalar);
      return out_scalar;
    });
  }
};

// tensor X -> string
template <typename SrcType>
struct TensorCaster<SrcType, std::string> {
  void Cast(const Tensor& in, Tensor& out, const TensorShape& shape) const {
    using DstType = std::string;
    const auto shape_size = shape.Size();
    const auto in_data = in.DataAsSpan<SrcType>();
    const auto out_data = out.MutableDataAsSpan<DstType>();
    for (int64_t i = 0; i < shape_size; ++i) {
      ScalarCaster<SrcType, DstType>{}.Cast(in_data[i], out_data[i]);
    }
  }
};

// tensor string -> X
template <typename DstType>
struct TensorCaster<std::string, DstType> {
  void Cast(const Tensor& in, Tensor& out, const TensorShape& shape) const {
    using SrcType = std::string;
    const auto shape_size = shape.Size();
    const auto in_data = in.DataAsSpan<SrcType>();
    const auto out_data = out.MutableDataAsSpan<DstType>();
    for (int64_t i = 0; i < shape_size; ++i) {
      ScalarCaster<SrcType, DstType>{}.Cast(in_data[i], out_data[i]);
    }
  }
};

// tensor float -> MLFloat16
template <>
struct TensorCaster<float, MLFloat16> {
  void Cast(const Tensor& in, Tensor& out, const TensorShape& shape) const {
    auto out_data = out.MutableData<MLFloat16>();
    auto shape_size = shape.Size();
    auto in_vector = ConstEigenVectorMap<float>(in.Data<float>(), shape_size);
    auto output_vector = EigenVectorMap<Eigen::half>(static_cast<Eigen::half*>(static_cast<void*>(out_data)),
                                                     shape_size);
    output_vector = in_vector.template cast<Eigen::half>();
  }
};

// tensor MLFloat16 -> float
template <>
struct TensorCaster<MLFloat16, float> {
  void Cast(const Tensor& in, Tensor& out, const TensorShape& shape) const {
    auto out_data = out.MutableData<float>();
    auto in_data = in.Data<MLFloat16>();
    auto shape_size = shape.Size();
#if defined(_M_AMD64)
    MlasConvertHalfToFloatBuffer(&in_data[0].val, out_data, shape_size);
#else
    auto in_vector = ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(in_data)),
                                                      shape_size);
    auto output_vector = EigenVectorMap<float>(out_data, shape_size);
    output_vector = in_vector.template cast<float>();
#endif
  }
};

class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
};

template <typename TSrc, typename TDst>
struct Dispatcher {
  void operator()(const Tensor& src, Tensor& dst, const TensorShape& shape) {
    TensorCaster<TSrc, TDst>{}.Cast(src, dst, shape);
  }
};

template <typename TSrc>
struct SrcDispatcher {
  void operator()(int32_t to, const Tensor& src, Tensor& dst, const TensorShape& shape) {
    using DstTypes = mp_remove_if_q<ImplementedDstTypes, mp_bind_front<std::is_same, TSrc>>;
    mp_apply<MLTypeCallDispatcher2, DstTypes> dispatcher{to};
    dispatcher.InvokeWithLeadingTemplateArgs<Dispatcher, TypeList<TSrc>>(src, dst, shape);
  }
};

Status Cast::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  if (shape.Size() == 0) {
    return Status::OK();
  }

  auto from = X->GetElementType();

  if (from == to_) {
    // will copy if X and Y have different buffers
    CopyCpuTensor(X, Y);
    return Status::OK();
  }

  mp_apply<MLTypeCallDispatcher2, ImplementedSrcTypes> dispatcher{from};
  dispatcher.Invoke<SrcDispatcher>(to_, *X, *Y, shape);

  return Status::OK();
}

const std::vector<MLDataType> castSrcTypeConstraints =
    mp_apply<BuildKernelDefConstraintsFunctor, ImplementedSrcTypes>{}();

const std::vector<MLDataType> castDstTypeConstraints =
    mp_apply<BuildKernelDefConstraintsFunctor, ImplementedDstTypes>{}();
}  // namespace



ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Cast,
    6,
    12,
    KernelDefBuilder()
        .TypeConstraint("T1", castSrcTypeConstraints)
        .TypeConstraint("T2", castDstTypeConstraints)
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

ONNX_CPU_OPERATOR_KERNEL(
    Cast,
    13,
    KernelDefBuilder()
        .TypeConstraint("T1", castSrcTypeConstraints)
        .TypeConstraint("T2", castDstTypeConstraints)
        .MayInplace(0, 0),  // allocation planner will check input and output sizes match before inplacing
    Cast);

}  // namespace onnxruntime
