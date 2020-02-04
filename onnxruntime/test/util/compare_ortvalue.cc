// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/compare_ortvalue.h"
#include <cmath>
#include <sstream>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4018) /*'expression' : signed/unsigned mismatch */
#pragma warning(disable : 4065) /*switch statement contains 'default' but no 'case' labels*/
#pragma warning(disable : 4100)
#pragma warning(disable : 4146) /*unary minus operator applied to unsigned type, result still unsigned*/
#pragma warning(disable : 4244) /*'conversion' conversion from 'type1' to 'type2', possible loss of data*/
#pragma warning(disable : 4251) /*'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'*/
#pragma warning(disable : 4267) /*'var' : conversion from 'size_t' to 'type', possible loss of data*/
#pragma warning(disable : 4305) /*'identifier' : truncation from 'type1' to 'type2'*/
#pragma warning(disable : 4307) /*'operator' : integral constant overflow*/
#pragma warning(disable : 4309) /*'conversion' : truncation of constant value*/
#pragma warning(disable : 4334) /*'operator' : result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)*/
#pragma warning(disable : 4355) /*'this' : used in base member initializer list*/
#pragma warning(disable : 4506) /*no definition for inline function 'function'*/
#pragma warning(disable : 4800) /*'type' : forcing value to bool 'true' or 'false' (performance warning)*/
#pragma warning(disable : 4996) /*The compiler encountered a deprecated declaration.*/
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif


#include "core/graph/onnx_protobuf.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "Eigen/Core"
#include "Eigen/src/Core/arch/Default/Half.h"

using namespace onnxruntime;

#if (!EIGEN_VERSION_AT_LEAST(3, 3, 6))
namespace Eigen {
namespace half_impl {
using __half_raw = ::Eigen::half_impl::__half;
}
}  // namespace Eigen
#endif

namespace {

template <typename T>
bool IsResultCloselyMatch(const T& outvalue, const T& expected_value, const double diff, const double tol) {
  if (diff > tol) return false;
  if (std::isnan(diff) && !(std::isnan(outvalue) && std::isnan(expected_value)) &&
      !(std::isinf(outvalue) && std::isinf(expected_value)))
    return false;
  return true;
}

template <typename FLOAT_TYPE>
std::pair<COMPARE_RESULT, std::string> CompareFloatResult(const Tensor& outvalue, const Tensor& expected_value,
                                                          double per_sample_tolerance,
                                                          double relative_per_sample_tolerance, bool post_processing) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const FLOAT_TYPE* expected_output = expected_value.template Data<FLOAT_TYPE>();
  const FLOAT_TYPE* real_output = outvalue.template Data<FLOAT_TYPE>();
  std::pair<COMPARE_RESULT, std::string> res = std::make_pair(COMPARE_RESULT::SUCCESS, "");
  double max_diff = 0;
  size_t diff_count = 0;
  for (size_t di = 0; di != size1; ++di) {
    const double real_value =
        post_processing ? std::max<double>(0.0, std::min<double>(255.0, real_output[di])) : real_output[di];
    const double diff = std::fabs(expected_output[di] - real_value);
    const double tol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected_output[di]);
    if (!IsResultCloselyMatch<double>(real_value, expected_output[di], diff, tol)) {
      res.first = COMPARE_RESULT::RESULT_DIFFERS;
      // update error message if this is a larger diff
      if (diff > max_diff || (std::isnan(diff) && !std::isnan(max_diff))) {
        int64_t expected_int = 0;
        int64_t real_int = 0;
        memcpy(&expected_int, &expected_output[di], sizeof(FLOAT_TYPE));
        memcpy(&real_int, &real_output[di], sizeof(FLOAT_TYPE));

        std::ostringstream oss;
        oss << std::hex << "expected " << expected_output[di] << " (" << expected_int << "), got " << real_value << " ("
            << real_int << ")"
            << ", diff: " << diff << ", tol=" << tol << ".";
        res.second = oss.str();
        max_diff = diff;
      }
      ++diff_count;
    }
  }

  if (res.first == COMPARE_RESULT::SUCCESS) return res;

  std::ostringstream oss;
  oss << res.second << " " << diff_count << " of " << size1 << " differ";
  res.second = oss.str();
  return res;
}

template <typename T>
std::pair<COMPARE_RESULT, std::string> IsResultExactlyMatch(const Tensor& outvalue, const Tensor& expected_value) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const T* expected_output = expected_value.template Data<T>();
  const T* real_output = outvalue.template Data<T>();
  for (size_t di = 0; di != size1; ++di) {
    if (expected_output[di] != real_output[di]) {
      std::ostringstream oss;
      oss << "expected " << expected_output[di] << ", got " << real_output[di];
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

std::pair<COMPARE_RESULT, std::string> CompareFloat16Result(const Tensor& outvalue, const Tensor& expected_value,
                                                            double per_sample_tolerance,
                                                            double relative_per_sample_tolerance,
                                                            bool post_processing) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const MLFloat16* expected_output = expected_value.template Data<MLFloat16>();
  const MLFloat16* real_output = outvalue.template Data<MLFloat16>();
  for (size_t di = 0; di != size1; ++di) {
    float expected = Eigen::half_impl::half_to_float(Eigen::half_impl::__half_raw(expected_output[di].val));
    float real = Eigen::half_impl::half_to_float(Eigen::half_impl::__half_raw(real_output[di].val));
    real = post_processing ? std::max(0.0f, std::min(255.0f, real)) : real;
    const double diff = std::fabs(expected - real);
    const double rtol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected);
    if (!IsResultCloselyMatch<float>(real, expected, diff, rtol)) {
      std::ostringstream oss;
      oss << "expected " << expected << ", got " << real << ", diff: " << diff << ", tol=" << rtol;

      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

std::pair<COMPARE_RESULT, std::string> CompareBFloat16Result(const Tensor& outvalue, const Tensor& expected_value,
                                                             double per_sample_tolerance,
                                                             double relative_per_sample_tolerance,
                                                             bool post_processing) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const BFloat16* expected_output = expected_value.template Data<BFloat16>();
  const BFloat16* real_output = outvalue.template Data<BFloat16>();
  for (size_t di = 0; di != size1; ++di) {
    float expected = expected_output[di].ToFloat();
    float real = real_output[di].ToFloat();
    real = post_processing ? std::max(0.0f, std::min(255.0f, real)) : real;
    const double diff = std::fabs(expected - real);
    const double rtol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected);
    if (!IsResultCloselyMatch<float>(real, expected, diff, rtol)) {
      std::ostringstream oss;
      oss << "expected " << expected << ", got " << real << ", diff: " << diff << ", tol=" << rtol;

      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

std::pair<COMPARE_RESULT, std::string> CompareTwoTensors(const Tensor& outvalue, const Tensor& expected_tensor,
                                                         double per_sample_tolerance,
                                                         double relative_per_sample_tolerance, bool post_processing) {
  if (expected_tensor.Shape() != outvalue.Shape()) {
    std::ostringstream oss;
    oss << "shape mismatch, expect " << expected_tensor.Shape().ToString() << " got " << outvalue.Shape().ToString();
    return std::make_pair(COMPARE_RESULT::SHAPE_MISMATCH, oss.str());
  }
  if (outvalue.IsDataType<float>()) {
    return CompareFloatResult<float>(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance,
                                     post_processing);
  } else if (outvalue.IsDataType<double>()) {
    return CompareFloatResult<double>(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance,
                                      post_processing);
  } else if (outvalue.IsDataTypeString()) {
    return IsResultExactlyMatch<std::string>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<uint8_t>()) {
    return IsResultExactlyMatch<uint8_t>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<int8_t>()) {
    return IsResultExactlyMatch<int8_t>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<uint16_t>()) {
    return IsResultExactlyMatch<uint16_t>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<int16_t>()) {
    return IsResultExactlyMatch<int16_t>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<uint32_t>()) {
    return IsResultExactlyMatch<uint32_t>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<int32_t>()) {
    return IsResultExactlyMatch<int32_t>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<uint64_t>()) {
    return IsResultExactlyMatch<uint64_t>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<int64_t>()) {
    return IsResultExactlyMatch<int64_t>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<bool>()) {
    return IsResultExactlyMatch<bool>(outvalue, expected_tensor);
  } else if (outvalue.IsDataType<MLFloat16>()) {
    return CompareFloat16Result(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance,
                                post_processing);
  } else if (outvalue.IsDataType<BFloat16>()) {
    return CompareBFloat16Result(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance,
                                 post_processing);
  } else {
    return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "");
  }
}
template <typename T>
std::pair<COMPARE_RESULT, std::string> CompareSeqOfMapToFloat(const T& real_output_vector, const T& expected_value,
                                                              double per_sample_tolerance,
                                                              double relative_per_sample_tolerance,
                                                              bool post_processing) {
  if (real_output_vector.size() != expected_value.size()) {
    std::ostringstream oss;
    oss << "vector size mismatch, expected " << expected_value.size() << ", got " << real_output_vector.size();
    return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
  }
  for (size_t i = 0; i != real_output_vector.size(); ++i) {
    const auto& expected_map = expected_value[i];
    // compare if expected_map equals real_output_vector[i]
    if (real_output_vector[i].size() != expected_map.size()) {
      std::ostringstream oss;
      oss << "map size mismatch, expected " << expected_map.size() << ", got " << real_output_vector[i].size();
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }

    for (const auto& real_output_key_value_pair : real_output_vector[i]) {
      auto expected_key_value_pair = expected_map.find(real_output_key_value_pair.first);
      if (expected_key_value_pair == expected_map.end()) {
        return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, "");
      }
      const double real = post_processing
                              ? std::max<double>(0.0, std::min<double>(255.0, real_output_key_value_pair.second))
                              : real_output_key_value_pair.second;
      const double diff = std::fabs(expected_key_value_pair->second - real);
      const double rtol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected_key_value_pair->second);
      if (!IsResultCloselyMatch<double>(real, expected_key_value_pair->second, diff, rtol)) {
        std::ostringstream oss;
        oss << "expected " << expected_key_value_pair->second << ", got " << real << ", diff: " << diff
            << ", tol=" << rtol;
        return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
      }
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

const char* ElementTypeToString(MLDataType type) {
  return DataTypeImpl::ToString(type);
}

// The expected_shape could contain unknown dimensions, but the real_shape cannot
bool AreShapesEqual(const std::vector<int64_t>& real_shape, const ::ONNX_NAMESPACE::TensorShapeProto& expected_shape) {
  const int len = expected_shape.dim_size();
  if (len < 0) return false;
  if (real_shape.size() != static_cast<size_t>(len)) return false;
  for (int i = 0; i != len; ++i) {
    const auto& dim = expected_shape.dim(i);
    switch (dim.value_case()) {
      case ONNX_NAMESPACE::TensorShapeProto::Dimension::kDimValue:
        if (dim.dim_value() != real_shape[i]) return false;
        break;
      case ONNX_NAMESPACE::TensorShapeProto::Dimension::kDimParam:
        // symbolic shape, cannot validate it right now, assume it matches every thing
        // fall through
      case ONNX_NAMESPACE::TensorShapeProto::Dimension::VALUE_NOT_SET:
        // Value not set is treated as can not be validated
        continue;
        break;
      // This is for unlikely case when we add new oneof value
      default:
        assert(false);
        break;
    }
  }
  return true;
}

template <typename T>
std::ostringstream& VectorToString(const std::vector<T>& input, std::ostringstream& oss) {
  size_t len = input.size();
  oss << "[";
  if (len > 0) {
    oss << input[0];
    for (size_t i = 1; i != len; ++i) {
      oss << ", " << input[i];
    }
  }
  oss << "]";
  return oss;
}

}  // namespace

namespace onnxruntime {
std::pair<COMPARE_RESULT, std::string> CompareOrtValue(const OrtValue& o, const OrtValue& expected_mlvalue,
                                                       double per_sample_tolerance,
                                                       double relative_per_sample_tolerance, bool post_processing) {
  if (o.IsTensor() != expected_mlvalue.IsTensor() || o.Type() != expected_mlvalue.Type()) {
    return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
  }
  if (!o.IsTensor()) {
    if (o.Type() == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
      return CompareSeqOfMapToFloat(o.Get<VectorMapInt64ToFloat>(), expected_mlvalue.Get<VectorMapInt64ToFloat>(),
                                    per_sample_tolerance, relative_per_sample_tolerance, post_processing);
    }
    if (o.Type() == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
      return CompareSeqOfMapToFloat(o.Get<VectorMapStringToFloat>(), expected_mlvalue.Get<VectorMapStringToFloat>(),
                                    per_sample_tolerance, relative_per_sample_tolerance, post_processing);
    }
    return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "");
  }
  const Tensor& outvalue = o.Get<Tensor>();
  const Tensor& expected_tensor = expected_mlvalue.Get<Tensor>();
  if (outvalue.DataType() != expected_tensor.DataType()) {
    std::ostringstream oss;
    oss << "expect " << ElementTypeToString(expected_tensor.DataType()) << " got "
        << ElementTypeToString(outvalue.DataType());
    return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, oss.str());
  }
  return CompareTwoTensors(outvalue, expected_tensor, per_sample_tolerance, relative_per_sample_tolerance,
                           post_processing);
}

std::pair<COMPARE_RESULT, std::string> VerifyValueInfo(const ONNX_NAMESPACE::ValueInfoProto& v, const Ort::Value& o) {
  if (!v.has_type()) return std::make_pair(COMPARE_RESULT::SUCCESS, "");
  if (v.type().has_tensor_type()) {
    if (!o.IsTensor()) {
      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
    }

    ::ONNX_NAMESPACE::TypeProto_Tensor t = v.type().tensor_type();
    // below code doesn't work
    // if (((TensorTypeBase*)o.Type())->GetElementType() != DataTypeImpl::ElementTypeFromProto(t.elem_type())) {
    //	return COMPARE_RESULT::TYPE_MISMATCH;
    //}
    auto info = o.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType real_type = info.GetElementType();
    ONNXTensorElementDataType expected_type = onnxruntime::utils::CApiElementTypeFromProtoType(t.elem_type());
    if (real_type != expected_type) {
      std::ostringstream oss;
      oss << "expect " << ElementTypeToString((MLDataType)expected_type) << " got "
          << ElementTypeToString((MLDataType)real_type);

      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, oss.str());
    }
    std::vector<int64_t> shape = info.GetShape();
    const auto& tensor_shape_proto = t.shape();
    if (!AreShapesEqual(shape, tensor_shape_proto)) {
      std::ostringstream oss;
      oss << "Tensor shape mismatch, model file expects '";
      if (tensor_shape_proto.dim_size() == 0) {
        oss << "(unknown)";
      } else {
        oss << tensor_shape_proto;
      }
      oss << "', real output is ";
      VectorToString(shape, oss);
      return std::make_pair(COMPARE_RESULT::SHAPE_MISMATCH, oss.str());
    }
  } else {
    // Cannot do this check for tensor type.
    // For tensor type, o.Type() is TensorTypeBase*, but p points to a subclass of TensorTypeBase
    auto p = DataTypeImpl::TypeFromProto(v.type());
    if (((OrtValue*)(const OrtValue*)o)->Type() != p) {
      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}
}  // namespace onnxruntime
