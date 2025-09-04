// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// Licensed under the MIT License.

#include "compare_ortvalue.h"

#include <cmath>
#include <sstream>
#include <mutex>
#include <thread>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-result"
// cmake/external/eigen/Eigen/src/Core/arch/NEON/PacketMath.h:1633:9:
// error: ‘void* memcpy(void*, const void*, size_t)’ copying an object of non-trivial type ‘Eigen::internal::Packet4c’
// {aka ‘struct Eigen::internal::eigen_packet_wrapper<int, 2>’} from an array of ‘const int8_t’
// {aka ‘const signed char’} [-Werror=class-memaccess]
#ifdef HAS_CLASS_MEMACCESS
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#endif
#include <google/protobuf/message_lite.h>
#include <Eigen/Core>
#include <Eigen/src/Core/arch/Default/Half.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "core/framework/utils.h"
#include "core/graph/onnx_protobuf.h"
#include <core/session/onnxruntime_cxx_api.h>
#include "core/common/common.h"

using namespace onnxruntime;

#if (!EIGEN_VERSION_AT_LEAST(3, 3, 6))
namespace Eigen {
namespace half_impl {
using __half_raw = ::Eigen::half_impl::__half;
}
}  // namespace Eigen
#endif

#define TEST_RETURN_IF_NOT(condition, compare_result, ...)                                                    \
  if (!(condition)) {                                                                                         \
    return std::make_pair(compare_result, ::onnxruntime::MakeString(ORT_WHERE.ToString(), " ", __VA_ARGS__)); \
  }

#define TEST_RETURN_IF_ERROR(stmt, ...)                                                                                \
  {                                                                                                                    \
    auto result_pair = (stmt);                                                                                         \
    if (result_pair.first != COMPARE_RESULT::SUCCESS) {                                                                \
      result_pair.second = ::onnxruntime::MakeString(ORT_WHERE.ToString(), " ", __VA_ARGS__, " ", result_pair.second); \
      return result_pair;                                                                                              \
    }                                                                                                                  \
  }

namespace {

const char* TensorElementTypeToString(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      return "undefined";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "complex64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "complex128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      return "float8e4m3fn";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      return "float8e4m3fnuz";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      return "float8e5m2";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      return "float8e5m2fnuz";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      return "uint4";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      return "int4";
    default:
      return "unknown";
  }
}

std::string TensorShapeToString(VectorInt64 shape) {
  std::string result;

  result.append("{");
  bool first = true;
  for (auto dim : shape) {
    if (!first) {
      result.append(",");
    }

    result.append(std::to_string(dim));
    first = false;
  }
  result.append("}");

  return result;
}

#define CASE_TYPE(X)                             \
  case ONNX_NAMESPACE::TensorProto_DataType_##X: \
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_##X;

ONNXTensorElementDataType CApiElementTypeFromProtoType(int type) {
  switch (type) {
    CASE_TYPE(FLOAT)
    CASE_TYPE(UINT8)
    CASE_TYPE(INT8)
    CASE_TYPE(UINT16)
    CASE_TYPE(INT16)
    CASE_TYPE(INT32)
    CASE_TYPE(INT64)
    CASE_TYPE(STRING)
    CASE_TYPE(BOOL)
    CASE_TYPE(FLOAT16)
    CASE_TYPE(DOUBLE)
    CASE_TYPE(UINT32)
    CASE_TYPE(UINT64)
    CASE_TYPE(COMPLEX64)
    CASE_TYPE(COMPLEX128)
    CASE_TYPE(BFLOAT16)
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_TYPE(FLOAT8E4M3FN)
    CASE_TYPE(FLOAT8E4M3FNUZ)
    CASE_TYPE(FLOAT8E5M2)
    CASE_TYPE(FLOAT8E5M2FNUZ)
#endif
    CASE_TYPE(UINT4)
    CASE_TYPE(INT4)
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

//#if defined(__aarch64__) && defined(__linux__)
template <typename T>
std::pair<COMPARE_RESULT, std::string> CheckCosineSimilarity(const Ort::Value& outvalue, const Ort::Value& expected_value) {
  size_t element_count = expected_value.GetTensorTypeAndShapeInfo().GetElementCount();
  const T* expected_output = expected_value.GetTensorData<T>();
  const T* real_output = outvalue.GetTensorData<T>();
  std::pair<COMPARE_RESULT, std::string> res = std::make_pair(COMPARE_RESULT::SUCCESS, "");
  const T cosine_similarity_threshold = 0.99f;

  T dot = 0.0f, denom_a = 0.0f, denom_b = 0.0f;
  for (size_t i = 0u; i < element_count; ++i) {
    if (isnan(expected_output[i]) && isnan(real_output[i]))
      continue;
    if (isinf(expected_output[i]) && isinf(real_output[i]))
      continue;
    dot += expected_output[i] * real_output[i];
    denom_a += expected_output[i] * expected_output[i];
    denom_b += real_output[i] * real_output[i];
  }

  T cos_factor = abs(dot / (sqrt(denom_a) * sqrt(denom_b)));
  if (cos_factor < cosine_similarity_threshold) {
    res.first = COMPARE_RESULT::RESULT_DIFFERS;
    std::ostringstream oss;
    oss << std::hex << "results differed, cosine similarity factor is " << cos_factor << ".";
    res.second = oss.str();
  }
  return res;
}

template <typename T>
std::pair<COMPARE_RESULT, std::string> CheckCloseMatch(const Ort::Value& outvalue, const Ort::Value& expected_value) {
  size_t element_count = expected_value.GetTensorTypeAndShapeInfo().GetElementCount();
  const T* expected_output = expected_value.Data<T>();
  const T* real_output = outvalue.Data<T>();
  const T close_match_threshold = 1.0;

  for (size_t i = 0; i != element_count; ++i) {
    const T diff = expected_output[i] - real_output[i];
    if (std::fabs(diff) > close_match_threshold) {
      std::ostringstream oss;
      oss << "expected " << expected_output[i] << ", got " << real_output[i];
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}
//#endif
/**
 * @brief Check if two values are closely matched with given tolerance.

 * Definition of closely match:
 * > If any of actual_value and expected_value is nan, actual_value and expected_value must both be nan.
 * > If any of actual_value and expected_value is inf, then actual_value and expected_value
 *   must both be inf with same sign.
 * > Otherwise, diff <= tol.

 * @param actual_value The value to be checked.
 * @param expected_value The baseline value used to check against.
 * @param diff The absolute difference calculated by the caller from actual_value and expected_value.
 * @param tol The absolute tolerance.
 * @return True when closely matched; False otherwise.
 */
template <typename T>
bool IsResultCloselyMatch(const T& actual_value, const T& expected_value, const double diff, const double tol) {
  if (std::isnan(actual_value) || std::isnan(expected_value))
    return std::isnan(actual_value) && std::isnan(expected_value);  // not possible both are not nan if diff is nan.

  if (std::isinf(actual_value) || std::isinf(expected_value)) {
    if (std::isinf(actual_value) && std::isinf(expected_value))
      return (actual_value > 0 && expected_value > 0) || (actual_value < 0 && expected_value < 0);
    else
      return false;
  }

  return (diff <= tol);
}

template <typename FLOAT_TYPE>
std::pair<COMPARE_RESULT, std::string> CompareFloatResult(const Ort::ConstValue& actual, const Ort::ConstValue& expected,
                                                          double per_sample_tolerance,
                                                          double relative_per_sample_tolerance, bool post_processing) {
  const FLOAT_TYPE* actual_output = actual.IsSparseTensor() ? actual.GetSparseTensorValues<FLOAT_TYPE>() : actual.GetTensorData<FLOAT_TYPE>();

  const FLOAT_TYPE* expected_output = expected.IsSparseTensor() ? 
                                      expected.GetSparseTensorValues<FLOAT_TYPE>() : expected.GetTensorData<FLOAT_TYPE>();

  size_t element_count = 0;
  if (actual.IsSparseTensor()) {
    auto values_shape = actual.GetSparseTensorValuesTypeAndShapeInfo().GetShape();
    element_count = values_shape.size() > 0 ? values_shape[0] : 0; // non-zeros
  } else {
    element_count = expected.GetTensorTypeAndShapeInfo().GetElementCount();
  }

  std::pair<COMPARE_RESULT, std::string> res = std::make_pair(COMPARE_RESULT::SUCCESS, "");
  double max_diff = 0;
  size_t diff_count = 0;
  for (size_t i = 0; i != element_count; ++i) {
    const double real_value =
        post_processing ? std::max<double>(0.0, std::min<double>(255.0, actual_output[i])) : actual_output[i];
    const double diff = std::fabs(expected_output[i] - real_value);
    const double tol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected_output[i]);
    if (!IsResultCloselyMatch<double>(real_value, expected_output[i], diff, tol)) {
      res.first = COMPARE_RESULT::RESULT_DIFFERS;
      // update error message if this is a larger diff
      if (diff > max_diff || (std::isnan(diff) && !std::isnan(max_diff))) {
        int64_t expected_int = 0;
        int64_t real_int = 0;
        memcpy(&expected_int, &expected_output[i], sizeof(FLOAT_TYPE));
        memcpy(&real_int, &actual_output[i], sizeof(FLOAT_TYPE));

        std::ostringstream oss;
        oss << std::hex << "expected " << expected_output[i] << " (" << expected_int << "), got " << real_value << " ("
            << real_int << ")"
            << ", diff: " << diff << ", tol=" << tol << std::dec << " idx=" << i << ".";
        res.second = oss.str();
        max_diff = diff;
      }
      ++diff_count;
    }
  }

  if (res.first == COMPARE_RESULT::SUCCESS) return res;

  std::ostringstream oss;
  oss << res.second << " " << diff_count << " of " << element_count << " differ";
  res.second = oss.str();
  return res;
}

template <typename T>
std::pair<COMPARE_RESULT, std::string> IsResultExactlyMatch(const Ort::ConstValue& outvalue, const Ort::ConstValue& expected_value) {
  size_t element_count = expected_value.GetTensorTypeAndShapeInfo().GetElementCount();
  const T* expected_output = expected_value.GetTensorData<T>();
  const T* real_output = outvalue.GetTensorData<T>();
  for (size_t i = 0; i != element_count; ++i) {
    if (expected_output[i] != real_output[i]) {
      std::ostringstream oss;
      oss << "expected " << expected_output[i] << ", got " << real_output[i];
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

/*
template <>
std::pair<COMPARE_RESULT, std::string> IsResultExactlyMatch<Int4x2>(const Tensor& outvalue,
                                                                    const Tensor& expected_value) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const Int4x2* expected_output = expected_value.Data<Int4x2>();
  const Int4x2* real_output = outvalue.Data<Int4x2>();
  for (size_t di = 0; di != size1; ++di) {
    size_t r = di >> 1;
    size_t c = di & 0x1;

    if (expected_output[r].GetElem(c) != real_output[r].GetElem(c)) {
      std::ostringstream oss;
      oss << "expected " << expected_output[r].GetElem(c) << ", got " << real_output[r].GetElem(c);
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

template <>
std::pair<COMPARE_RESULT, std::string> IsResultExactlyMatch<UInt4x2>(const Tensor& outvalue,
                                                                     const Tensor& expected_value) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const UInt4x2* expected_output = expected_value.Data<UInt4x2>();
  const UInt4x2* real_output = outvalue.Data<UInt4x2>();
  for (size_t di = 0; di != size1; ++di) {
    size_t r = di >> 1;
    size_t c = di & 0x1;

    if (expected_output[r].GetElem(c) != real_output[r].GetElem(c)) {
      std::ostringstream oss;
      oss << "expected " << expected_output[r].GetElem(c) << ", got " << real_output[r].GetElem(c);
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}
*/

std::pair<COMPARE_RESULT, std::string> CompareFloat16Result(const Ort::ConstValue& outvalue, const Ort::ConstValue& expected_value,
                                                            double per_sample_tolerance,
                                                            double relative_per_sample_tolerance,
                                                            bool post_processing) {
  size_t element_count = expected_value.GetTensorTypeAndShapeInfo().GetElementCount();
  const MLFloat16* expected_output = expected_value.GetTensorData<MLFloat16>();
  const MLFloat16* real_output = outvalue.GetTensorData<MLFloat16>();
  std::ostringstream oss;
  COMPARE_RESULT result = COMPARE_RESULT::SUCCESS;
  for (size_t i = 0; i != element_count; ++i) {
    float expected = Eigen::half_impl::half_to_float(Eigen::half_impl::__half_raw(expected_output[i].val));
    float real = Eigen::half_impl::half_to_float(Eigen::half_impl::__half_raw(real_output[i].val));
    real = post_processing ? std::max(0.0f, std::min(255.0f, real)) : real;
    const double diff = std::fabs(expected - real);
    const double rtol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected);
    if (!IsResultCloselyMatch<float>(real, expected, diff, rtol)) {
      oss << "idx: " << i << ", expected " << expected << ", got " << real << ", diff: " << diff << ", tol=" << rtol << "\n";
      result = COMPARE_RESULT::RESULT_DIFFERS;
    }
  }
  return std::make_pair(result, oss.str());
}

std::pair<COMPARE_RESULT, std::string> CompareBFloat16Result(const Ort::ConstValue& outvalue, const Ort::ConstValue& expected_value,
                                                             double per_sample_tolerance,
                                                             double relative_per_sample_tolerance,
                                                             bool post_processing) {
  size_t element_count = expected_value.GetTensorTypeAndShapeInfo().GetElementCount();
  const BFloat16* expected_output = expected_value.GetTensorData<BFloat16>();
  const BFloat16* real_output = outvalue.GetTensorData<BFloat16>();
  for (size_t i = 0; i != element_count; ++i) {
    float expected = expected_output[i].ToFloat();
    float real = real_output[i].ToFloat();
    real = post_processing ? std::max(0.0f, std::min(255.0f, real)) : real;
    const double diff = std::fabs(expected - real);
    const double rtol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected);
    if (!IsResultCloselyMatch<float>(real, expected, diff, rtol)) {
      std::ostringstream oss;
      oss << "idx: " << i << ", expected " << expected << ", got " << real << ", diff: " << diff << ", tol=" << rtol << "\n";

      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

std::pair<COMPARE_RESULT, std::string> CompareSparseTensorIndices(const int64_t* real, const int64_t* expected, size_t num_element) {
  for (size_t i = 0; i != num_element; ++i) {
    if (real[i] != expected[i]) {
      std::ostringstream oss;
      oss << "expected " << expected[i] << ", got " << real[i];
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }
  }
  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

std::pair<COMPARE_RESULT, std::string> CompareTensorsOfOrtValues(const Ort::ConstValue& actual_value, const Ort::ConstValue& expected_value,
                                                                 double per_sample_tolerance,
                                                                 double relative_per_sample_tolerance, bool post_processing) {
  Ort::TensorTypeAndShapeInfo type_shape_info = actual_value.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType element_type = type_shape_info.GetElementType();
  auto shape = type_shape_info.GetShape();

  auto expected_shape = expected_value.GetTensorTypeAndShapeInfo().GetShape();

  if (shape != expected_shape) {
    std::ostringstream oss;
    oss << "shape mismatch, expect " << TensorShapeToString(expected_shape) << " got " << TensorShapeToString(shape);
    return std::make_pair(COMPARE_RESULT::SHAPE_MISMATCH, oss.str());
  }

#if defined(__aarch64__) && defined(__linux__)
  if (isnan(per_sample_tolerance) || isnan(per_sample_tolerance)) {
    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return CheckCosineSimilarity<float>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      return CheckCosineSimilarity<double>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      return CheckCloseMatch<uint8_t>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
      return CheckCloseMatch<int8_t>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
      return CheckCloseMatch<uint16_t>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
      return CheckCloseMatch<int16_t>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
      return CheckCloseMatch<uint32_t>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      return CheckCloseMatch<int32_t>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) {
      return CheckCloseMatch<uint64_t>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return CheckCloseMatch<int64_t>(actual_value, expected_value);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      return CheckCloseMatch<bool>(actual_value, expected_value);
    } else {
      return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "");
    }
  }
#endif

  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return CompareFloatResult<float>(actual_value, expected_value, per_sample_tolerance, relative_per_sample_tolerance,
                                     post_processing);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return CompareFloatResult<double>(actual_value, expected_value, per_sample_tolerance, relative_per_sample_tolerance,
                                      post_processing);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    return IsResultExactlyMatch<std::string>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    return IsResultExactlyMatch<uint8_t>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    return IsResultExactlyMatch<int8_t>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    return IsResultExactlyMatch<uint16_t>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
    return IsResultExactlyMatch<int16_t>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
    return IsResultExactlyMatch<uint32_t>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return IsResultExactlyMatch<int32_t>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) {
    return IsResultExactlyMatch<uint64_t>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return IsResultExactlyMatch<int64_t>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return IsResultExactlyMatch<bool>(actual_value, expected_value);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    return CompareFloat16Result(actual_value, expected_value, per_sample_tolerance, relative_per_sample_tolerance,
                                post_processing);
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    return CompareBFloat16Result(actual_value, expected_value, per_sample_tolerance, relative_per_sample_tolerance,
                                 post_processing);
  } else {
    // Note: Int4x2 and UInt4x2 are not exposed as public type, so we don't support that types.
    return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "");
  }
}

std::pair<COMPARE_RESULT, std::string> CompareMapToFloat(const Ort::ConstValue& actual, const Ort::ConstValue& expected,
                                                         double per_sample_tolerance,
                                                         double relative_per_sample_tolerance,
                                                         bool post_processing) {

  const Ort::ConstValue keys = actual.GetValue(0, Ort::AllocatorWithDefaultOptions()).GetConst();
  const Ort::ConstValue values = actual.GetValue(1, Ort::AllocatorWithDefaultOptions()).GetConst();
  size_t num_keys = keys.GetTensorTypeAndShapeInfo().GetElementCount();

  const Ort::ConstValue expected_keys = expected.GetValue(0, Ort::AllocatorWithDefaultOptions()).GetConst();
  const Ort::ConstValue expected_values = expected.GetValue(1, Ort::AllocatorWithDefaultOptions()).GetConst();
  size_t expected_num_keys = expected_keys.GetTensorTypeAndShapeInfo().GetElementCount();

  if (num_keys != expected_num_keys) {
    std::ostringstream oss;
    oss << "map size mismatch, expected " << expected_num_keys << ", got " << num_keys;
    return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
  }

  // Check keys of the map
  TEST_RETURN_IF_ERROR(CompareTensorsOfOrtValues(keys, expected_keys,
                                                 per_sample_tolerance, relative_per_sample_tolerance, post_processing),
                       "While comparing keys of the Map");

  // Check values of the map
  TEST_RETURN_IF_ERROR(CompareTensorsOfOrtValues(values, expected_values,
                                                 per_sample_tolerance, relative_per_sample_tolerance, post_processing),
                       "While comparing values of the Map");

  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
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

#if !defined(DISABLE_SPARSE_TENSORS)
std::pair<COMPARE_RESULT, std::string> CompareSparseTensorsOfOrtValues(const Ort::ConstValue& actual, const Ort::ConstValue& expected,
                                                                       double per_sample_tolerance, double relative_per_sample_tolerance,
                                                                       bool post_processing) {
  // Check dense shape
  Ort::TensorTypeAndShapeInfo type_shape_info = actual.GetTensorTypeAndShapeInfo();
  auto shape = actual.GetTensorTypeAndShapeInfo().GetShape();
  Ort::TensorTypeAndShapeInfo expected_type_shape_info = expected.GetTensorTypeAndShapeInfo();
  auto expected_shape = expected.GetTensorTypeAndShapeInfo().GetShape();

  if (shape != expected_shape) {
    std::ostringstream oss;
    oss << "dense shape mismatch, expect " << TensorShapeToString(expected_shape) << " got " << TensorShapeToString(shape);
    return std::make_pair(COMPARE_RESULT::SHAPE_MISMATCH, oss.str());
  }

  // Check sparsity format
  auto sparse_format = actual.GetSparseFormat();
  auto expected_sparse_format = expected.GetSparseFormat();

  TEST_RETURN_IF_NOT(sparse_format == expected_sparse_format, COMPARE_RESULT::TYPE_MISMATCH,
                     "Expected sparse format: ", expected_sparse_format,
                     " actual: ", sparse_format);

  // Check non-zero element counts
  auto vlaues_type_and_shape = actual.GetSparseTensorValuesTypeAndShapeInfo();
  auto values_shape = vlaues_type_and_shape.GetShape();
  auto expected_vlaues_type_and_shape = expected.GetSparseTensorValuesTypeAndShapeInfo();
  auto expected_values_shape = expected_vlaues_type_and_shape.GetShape();
  TEST_RETURN_IF_NOT(values_shape[0] == expected_values_shape[0], COMPARE_RESULT::TYPE_MISMATCH,
                     "Expected number of non-zero values: ", expected_values_shape[0],
                     " actual: ", values_shape[0]);

  // Check element type
  ONNXTensorElementDataType element_type = vlaues_type_and_shape.GetElementType();
  ONNXTensorElementDataType expected_element_type = expected_vlaues_type_and_shape.GetElementType();
  TEST_RETURN_IF_NOT(element_type == expected_element_type, COMPARE_RESULT::TYPE_MISMATCH,
                     "Expected type: ", TensorElementTypeToString(expected_element_type),
                     " actual: ", TensorElementTypeToString(element_type));

  // Check non-zero values
  TEST_RETURN_IF_ERROR(CompareTensorsOfOrtValues(actual, expected,
                                                 per_sample_tolerance, relative_per_sample_tolerance, post_processing),
                       "While comparing sparse values");

  // Check indices
  if (sparse_format == OrtSparseFormat::ORT_SPARSE_COO) {
    size_t num_indices;
    size_t expected_num_indices;

    const int64_t* indices = actual.GetSparseTensorIndicesData<int64_t>(OrtSparseIndicesFormat::ORT_SPARSE_COO_INDICES, num_indices);
    const int64_t* expected_indices = expected.GetSparseTensorIndicesData<int64_t>(OrtSparseIndicesFormat::ORT_SPARSE_COO_INDICES, expected_num_indices);

    TEST_RETURN_IF_NOT(num_indices == expected_num_indices, COMPARE_RESULT::SHAPE_MISMATCH,
                       "Expected number of index: ", expected_num_indices,
                       " actual: ", num_indices);

    TEST_RETURN_IF_ERROR(CompareSparseTensorIndices(indices, expected_indices, num_indices),
                         "Comparing COO indices");

  } else if (sparse_format == OrtSparseFormat::ORT_SPARSE_CSRC) {
    // Innder indices
    size_t num_inner_indices;
    size_t expected_num_inner_indices;
    const int64_t* inner_indices = actual.GetSparseTensorIndicesData<int64_t>(OrtSparseIndicesFormat::ORT_SPARSE_CSR_INNER_INDICES, num_inner_indices);
    const int64_t* expected_inner_indices = expected.GetSparseTensorIndicesData<int64_t>(OrtSparseIndicesFormat::ORT_SPARSE_CSR_INNER_INDICES, expected_num_inner_indices);

    TEST_RETURN_IF_NOT(num_inner_indices == expected_num_inner_indices, COMPARE_RESULT::SHAPE_MISMATCH,
                       "Expected number of inner index: ", expected_num_inner_indices,
                       " actual: ", num_inner_indices);

    
    TEST_RETURN_IF_ERROR(CompareSparseTensorIndices(inner_indices, expected_inner_indices, num_inner_indices),
                         "Comparing Csr(c) inner indices");

    // Outer indices
    size_t num_outer_indices;
    size_t expected_num_outer_indices;
    const int64_t* outer_indices = actual.GetSparseTensorIndicesData<int64_t>(OrtSparseIndicesFormat::ORT_SPARSE_CSR_OUTER_INDICES, num_outer_indices);
    const int64_t* expected_outer_indices = expected.GetSparseTensorIndicesData<int64_t>(OrtSparseIndicesFormat::ORT_SPARSE_CSR_OUTER_INDICES, expected_num_outer_indices);

    TEST_RETURN_IF_NOT(num_outer_indices == expected_num_outer_indices, COMPARE_RESULT::SHAPE_MISMATCH,
                       "Expected number of outer index: ", expected_num_outer_indices,
                       " actual: ", num_outer_indices);

    TEST_RETURN_IF_ERROR(CompareSparseTensorIndices(outer_indices, expected_outer_indices, num_outer_indices),
                         "Comparing Csr(c) outer indices");
  }

  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

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
std::pair<COMPARE_RESULT, std::string> CompareOrtValue(const OrtValue& actual_value, const OrtValue& expected_value,
                                                       double per_sample_tolerance,
                                                       double relative_per_sample_tolerance, bool post_processing) {
  const Ort::ConstValue output_mlvalue{&actual_value};
  const Ort::ConstValue expected_mlvalue{&expected_value};

  // Only following OrtValue types are supported:
  // - Tensor
  // - Sparse Tensor
  // - Sequence of Tensors
  // - Sequence of Maps (where map is either 'int64 to float' or 'string to float')

  ONNXType output_mlvalue_type;
  ONNXType expected_mlvalue_type;
  Ort::ThrowOnError(Ort::GetApi().GetValueType(output_mlvalue, &output_mlvalue_type));
  Ort::ThrowOnError(Ort::GetApi().GetValueType(expected_mlvalue, &expected_mlvalue_type));

  if (output_mlvalue_type != expected_mlvalue_type) {
    return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
  }

  if (output_mlvalue_type == ONNX_TYPE_TENSOR) { // Tensor
    ONNXTensorElementDataType element_type = output_mlvalue.GetTensorTypeAndShapeInfo().GetElementType();
    ONNXTensorElementDataType expected_element_type = expected_mlvalue.GetTensorTypeAndShapeInfo().GetElementType();

    if (element_type != expected_element_type) {
      std::ostringstream oss;
      oss << "expect " << TensorElementTypeToString(expected_element_type) << " got "
          << TensorElementTypeToString(element_type);
      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, oss.str());
    }

    return CompareTensorsOfOrtValues(output_mlvalue, expected_mlvalue, per_sample_tolerance, relative_per_sample_tolerance,
                                     post_processing);

  } else if (output_mlvalue_type == ONNX_TYPE_SPARSETENSOR) { // Sparse tensor
#if !defined(DISABLE_SPARSE_TENSORS)
    TEST_RETURN_IF_NOT(expected_mlvalue.IsSparseTensor(), COMPARE_RESULT::TYPE_MISMATCH,
                       "SparseTensor is not expected as output");
    TEST_RETURN_IF_ERROR(CompareSparseTensorsOfOrtValues(output_mlvalue, expected_mlvalue,
                                              per_sample_tolerance, relative_per_sample_tolerance,
                                              post_processing),
                         "while comaring sparse tensors");
#endif
    return std::make_pair(COMPARE_RESULT::SUCCESS, "");

  } else if (output_mlvalue_type == ONNX_TYPE_SEQUENCE) { // Sequence
    size_t num_elements = output_mlvalue.GetCount();
    size_t expected_num_elements = expected_mlvalue.GetCount();

    if (num_elements != expected_num_elements) {
      std::ostringstream oss;
      oss << "expected tensor/element count in the sequence: " << expected_num_elements << " got "
          << num_elements;
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }

    ONNXTensorElementDataType element_type = output_mlvalue.GetTensorTypeAndShapeInfo().GetElementType();
    ONNXTensorElementDataType expected_element_type = expected_mlvalue.GetTensorTypeAndShapeInfo().GetElementType();

    if (element_type != expected_element_type) {
      std::ostringstream oss;
      oss << "expect " << TensorElementTypeToString(expected_element_type) << " got "
          << TensorElementTypeToString(element_type);
      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, oss.str());
    }

    for (int i = 0; i < expected_num_elements; ++i) {
      const Ort::ConstValue actual_ort_value = output_mlvalue.GetValue(i, Ort::AllocatorWithDefaultOptions()).GetConst();
      const Ort::ConstValue expect_ort_value = expected_mlvalue.GetValue(i, Ort::AllocatorWithDefaultOptions()).GetConst();

      ONNXType onnx_type;
      ONNXType expected_onnx_type;
      Ort::ThrowOnError(Ort::GetApi().GetValueType(output_mlvalue, &onnx_type));
      Ort::ThrowOnError(Ort::GetApi().GetValueType(expected_mlvalue, &expected_onnx_type));

      if (onnx_type != expected_onnx_type) {
        return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
      }

      if (onnx_type == ONNX_TYPE_TENSOR) {  // Tensor type. It means the original OrtValue is sequence of tensors.
        auto res = CompareTensorsOfOrtValues(actual_ort_value, expect_ort_value, per_sample_tolerance, relative_per_sample_tolerance,
                                             post_processing);
        if (res.first != COMPARE_RESULT::SUCCESS) {
          return res;
        }
      } 
      else if (onnx_type == ONNX_TYPE_MAP) { // Map type. It means the original OrtValue is sequence of maps.
#if !defined(DISABLE_ML_OPS)
        const Ort::ConstValue key = actual_ort_value.GetValue(0, Ort::AllocatorWithDefaultOptions()).GetConst();
        const Ort::ConstValue value = actual_ort_value.GetValue(1, Ort::AllocatorWithDefaultOptions()).GetConst();
        ONNXTensorElementDataType key_element_type = key.GetTensorTypeAndShapeInfo().GetElementType();
        ONNXTensorElementDataType value_element_type = value.GetTensorTypeAndShapeInfo().GetElementType();

        if (key.IsTensor() &&
            value.IsTensor() &&
            (key_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || key_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) &&
            value_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          auto res = CompareMapToFloat(actual_ort_value, expect_ort_value,
                                       per_sample_tolerance, relative_per_sample_tolerance, post_processing);
          if (res.first != COMPARE_RESULT::SUCCESS) {
            return res;
          }
        } else {
          return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "Only sequence of maps ('int64 to float' or 'string to float') are supported.");
        }

#else
        return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "Sequence of maps is not supported in this build.");
#endif
      }
      else {
        return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "Only sequence of tensors/maps are supported.");
      }
    }

    return std::make_pair(COMPARE_RESULT::SUCCESS, "");
  } else {
    return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "Only tensor, sparse tensor, sequnce of tensors/maps are supported.");
  }
}

static std::pair<COMPARE_RESULT, std::string> CompareTensorOrtValueAndTensorTypeProto(const ONNX_NAMESPACE::TypeProto_Tensor& t,
                                                                                      const OrtValue* v) {
  // below code doesn't work
  // if (((TensorTypeBase*)o.Type())->GetElementType() != DataTypeImpl::ElementTypeFromProto(t.elem_type())) {
  //	return COMPARE_RESULT::TYPE_MISMATCH;
  //}

  Ort::ConstValue o(v);
  auto info = o.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType real_type = info.GetElementType();
  ONNXTensorElementDataType expected_type = CApiElementTypeFromProtoType(t.elem_type());
  if (real_type != expected_type) {
    std::ostringstream oss;
    oss << "expect " << TensorElementTypeToString(expected_type) << " got "
        << TensorElementTypeToString(real_type);

    return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, oss.str());
  }
  std::vector<int64_t> shape = info.GetShape();
  const auto& tensor_shape_proto = t.shape();
  if (!AreShapesEqual(shape, tensor_shape_proto)) {
    std::ostringstream oss;

    // same as utils::GetTensorShapeFromTensorShapeProto
    auto GetTensorShapeFromTensorShapeProto = [&](const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto) {
      const auto& dims = tensor_shape_proto.dim();
      std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
      for (int i = 0; i < dims.size(); ++i) {
        bool has_dim_value = dims[i].value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue;
        tensor_shape_vec[i] =
            has_dim_value ? dims[i].dim_value() : -1; /* symbolic dimensions are represented as -1 in onnxruntime*/
      }
      return tensor_shape_vec;
    };

    oss << "Tensor shape mismatch, model file expects '";
    if (tensor_shape_proto.dim_size() == 0) {
      oss << "(unknown)";
    } else {
      std::vector<int64_t> expected_shape = GetTensorShapeFromTensorShapeProto(tensor_shape_proto);
      VectorToString(expected_shape, oss);
    }
    oss << "', real output is ";
    VectorToString(shape, oss);
    return std::make_pair(COMPARE_RESULT::SHAPE_MISMATCH, oss.str());
  }

  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}

namespace {
template <typename T>
float ParseValueToFloat(T data_value) {
  return static_cast<float>(data_value);
}

template <>
float ParseValueToFloat(MLFloat16 data_value) {
  return Eigen::half_impl::half_to_float(Eigen::half_impl::__half_raw(data_value.val));
}

template <>
float ParseValueToFloat(float data_value) {
  // Covert float to half and then convert back to float to simulate rounding to half
  return ParseValueToFloat(MLFloat16(data_value));
}

template <typename RealT, typename ExpectT>
std::pair<COMPARE_RESULT, std::string> CompareFloat16WithFloatResult(const Tensor& outvalue,
                                                                     const Tensor& expected_value,
                                                                     double per_sample_tolerance,
                                                                     double relative_per_sample_tolerance) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const ExpectT* expected_output = expected_value.Data<ExpectT>();
  const RealT* real_output = outvalue.Data<RealT>();

  COMPARE_RESULT result = COMPARE_RESULT::SUCCESS;
  std::string error_msg = "";

  OrtThreadPoolParams to;
  to.thread_pool_size = 16;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);

  static double cost = 1;
  std::once_flag write_flag;
  concurrency::ThreadPool::TryParallelFor(
      tp.get(), size1, cost,
      [&error_msg, &result, &expected_output, &real_output, per_sample_tolerance,
       relative_per_sample_tolerance, size1, &write_flag](
          std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t di = begin; di != end && di < static_cast<std::ptrdiff_t>(size1); ++di) {
          float expected = ParseValueToFloat(expected_output[di]);
          float real = ParseValueToFloat(real_output[di]);
          const double diff = std::fabs(expected - real);
          const double rtol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected);
          if (!IsResultCloselyMatch<float>(real, expected, diff, rtol)) {
            std::ostringstream oss;
            oss << "idx: " << di << ", expected " << expected << ", got " << real
                << ", diff: " << diff << ", tol=" << rtol << "\n";
            std::call_once(write_flag, [&error_msg, &result, &oss]() {
              error_msg = oss.str();
              result = COMPARE_RESULT::RESULT_DIFFERS;
            });
            break;
          }
        }
      });

  return std::make_pair(result, error_msg);
}

}  // namespace

std::pair<COMPARE_RESULT, std::string> VerifyValueInfo(const ONNX_NAMESPACE::ValueInfoProto& v, const OrtValue* val_ptr) {
  Ort::ConstValue o{val_ptr};
  if (!v.has_type()) return std::make_pair(COMPARE_RESULT::SUCCESS, "");
  if (v.type().has_tensor_type()) {
    if (!o.IsTensor()) {
      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
    }

    ::ONNX_NAMESPACE::TypeProto_Tensor t = v.type().tensor_type();

    return CompareTensorOrtValueAndTensorTypeProto(t, o);
  } else if (v.type().has_sequence_type()) {
    // TODO: CXX API doesn't have IsTensorSequence() supported for Ort::Value
    // TODO: Repeat whatever we did for Tensor above in a loop ?
    return std::make_pair(COMPARE_RESULT::SUCCESS, "");
  } else if (v.type().has_optional_type()) {
    const auto& tp = v.type().optional_type().elem_type();

    if (tp.has_tensor_type() && !o.IsTensor()) {
      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
    }

    // For None, we do not have to validate anything against the ValueInfoProto.
    // If we have reached this point and are in possession of a None, we have
    // already ensured that the expected OrtValue is None as well.
    if (!o.HasValue()) {
      ::ONNX_NAMESPACE::TypeProto_Tensor t = tp.tensor_type();

      return CompareTensorOrtValueAndTensorTypeProto(t, o);
    }

    // TODO: Deal with sequences the same way we choose to deal with it
    // in the above else if()

  } else {
    // Cannot do this check for tensor/sequence of tensor type.
    // For tensor type, o.Type() is TensorTypeBase*, but p points to a subclass of TensorTypeBase
    // For sequences of tensor type, o.Type() is SequenceTensorTypeBase*, but p points to a subclass of SequenceTensorTypeBase
    //MLDataType p = DataTypeImpl::TypeFromProto(v.type());
    //MLDataType q = ((OrtValue*)(const OrtValue*)o)->Type();
    //if (q != p) {
    //  return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
    //}
  }

  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}
}  // namespace onnxruntime
