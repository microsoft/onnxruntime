// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/compare_ortvalue.h"

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

#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/framework/TensorSeq.h"
#include "core/graph/onnx_protobuf.h"
#include <core/session/onnxruntime_cxx_api.h>
#include "core/util/math.h"

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

const char* ElementTypeToString(MLDataType type) {
  return DataTypeImpl::ToString(type);
}

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
std::pair<COMPARE_RESULT, std::string> CompareFloatResult(const Tensor& outvalue, const Tensor& expected_value,
                                                          double per_sample_tolerance,
                                                          double relative_per_sample_tolerance, bool post_processing) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const FLOAT_TYPE* expected_output = expected_value.Data<FLOAT_TYPE>();
  const FLOAT_TYPE* real_output = outvalue.Data<FLOAT_TYPE>();
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
            << ", diff: " << diff << ", tol=" << tol << std::dec << " idx=" << di << ".";
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
  const T* expected_output = expected_value.Data<T>();
  const T* real_output = outvalue.Data<T>();
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
  const MLFloat16* expected_output = expected_value.Data<MLFloat16>();
  const MLFloat16* real_output = outvalue.Data<MLFloat16>();
  std::ostringstream oss;
  COMPARE_RESULT result = COMPARE_RESULT::SUCCESS;
  for (size_t di = 0; di != size1; ++di) {
    float expected = Eigen::half_impl::half_to_float(Eigen::half_impl::__half_raw(expected_output[di].val));
    float real = Eigen::half_impl::half_to_float(Eigen::half_impl::__half_raw(real_output[di].val));
    real = post_processing ? std::max(0.0f, std::min(255.0f, real)) : real;
    const double diff = std::fabs(expected - real);
    const double rtol = per_sample_tolerance + relative_per_sample_tolerance * std::fabs(expected);
    if (!IsResultCloselyMatch<float>(real, expected, diff, rtol)) {
      oss << "idx: " << di << "expected " << expected << ", got " << real << ", diff: " << diff << ", tol=" << rtol << "\n";
      result = COMPARE_RESULT::RESULT_DIFFERS;
    }
  }
  return std::make_pair(result, oss.str());
}

std::pair<COMPARE_RESULT, std::string> CompareBFloat16Result(const Tensor& outvalue, const Tensor& expected_value,
                                                             double per_sample_tolerance,
                                                             double relative_per_sample_tolerance,
                                                             bool post_processing) {
  const size_t size1 = static_cast<size_t>(expected_value.Shape().Size());
  const BFloat16* expected_output = expected_value.Data<BFloat16>();
  const BFloat16* real_output = outvalue.Data<BFloat16>();
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

#if !defined(DISABLE_SPARSE_TENSORS)
std::pair<COMPARE_RESULT, std::string> CompareSparseTensors(const SparseTensor& actual, const SparseTensor& expected,
                                                            double per_sample_tolerance, double relative_per_sample_tolerance,
                                                            bool post_processing) {
  TEST_RETURN_IF_NOT(actual.DataType() == expected.DataType(), COMPARE_RESULT::TYPE_MISMATCH,
                     "Expected type: ", ElementTypeToString(expected.DataType()),
                     " actual: ", ElementTypeToString(actual.DataType()));

  TEST_RETURN_IF_NOT(actual.DenseShape() == expected.DenseShape(), COMPARE_RESULT::SHAPE_MISMATCH,
                     "Expected dense shape: ", expected.DenseShape(),
                     " Actual: ", actual.DenseShape());

  TEST_RETURN_IF_NOT(actual.Format() == expected.Format(), COMPARE_RESULT::TYPE_MISMATCH,
                     "Expected sparse format", expected.Format(),
                     " actual: ", actual.Format());

  TEST_RETURN_IF_ERROR(CompareTwoTensors(actual.Values(), expected.Values(),
                                         per_sample_tolerance, relative_per_sample_tolerance, post_processing),
                       "While comparing sparse values");

  if (actual.Format() == SparseFormat::kCoo) {
    auto actual_view = actual.AsCoo();
    auto expected_view = expected.AsCoo();

    TEST_RETURN_IF_ERROR(CompareTwoTensors(actual_view.Indices(), expected_view.Indices(),
                                           per_sample_tolerance, relative_per_sample_tolerance, post_processing),
                         "Comparing COO indices");
  } else if (actual.Format() == SparseFormat::kCsrc) {
    auto actual_view = actual.AsCsr();
    auto expected_view = expected.AsCsr();
    TEST_RETURN_IF_ERROR(CompareTwoTensors(actual_view.Inner(), expected_view.Inner(),
                                           per_sample_tolerance, relative_per_sample_tolerance, post_processing),
                         "Comparing Csr(c) inner indices");
    TEST_RETURN_IF_ERROR(CompareTwoTensors(actual_view.Outer(), expected_view.Outer(),
                                           per_sample_tolerance, relative_per_sample_tolerance, post_processing),
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
std::pair<COMPARE_RESULT, std::string> CompareOrtValue(const OrtValue& o, const OrtValue& expected_mlvalue,
                                                       double per_sample_tolerance,
                                                       double relative_per_sample_tolerance, bool post_processing) {
  if (o.Type() != expected_mlvalue.Type()) {
    return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
  }
  if (o.IsTensor()) {
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
  } else if (o.IsSparseTensor()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    TEST_RETURN_IF_NOT(expected_mlvalue.IsSparseTensor(), COMPARE_RESULT::TYPE_MISMATCH,
                       "SparseTensor is not expected as output");
    TEST_RETURN_IF_ERROR(CompareSparseTensors(o.Get<SparseTensor>(), expected_mlvalue.Get<SparseTensor>(),
                                              per_sample_tolerance, relative_per_sample_tolerance,
                                              post_processing),
                         "while comaring sparse tensors");
#endif
    return std::make_pair(COMPARE_RESULT::SUCCESS, "");
  } else if (o.IsTensorSequence()) {
    auto& expected_tensor_seq = expected_mlvalue.Get<TensorSeq>();
    auto expected_tensor_count = expected_tensor_seq.Size();

    auto& actual_tensor_seq = o.Get<TensorSeq>();
    auto actual_tensor_count = actual_tensor_seq.Size();

    if (expected_tensor_count != actual_tensor_count) {
      std::ostringstream oss;
      oss << "expected tensor count in the sequence: " << expected_tensor_count << " got "
          << actual_tensor_count;
      return std::make_pair(COMPARE_RESULT::RESULT_DIFFERS, oss.str());
    }

    if (!expected_tensor_seq.IsSameDataType(actual_tensor_seq)) {
      std::ostringstream oss;
      oss << "expected tensor type in the sequence: " << expected_tensor_seq.DataType() << " got "
          << actual_tensor_seq.DataType();
      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, oss.str());
    }

    for (size_t i = 0; i < expected_tensor_count; ++i) {
      auto res = CompareTwoTensors(actual_tensor_seq.Get(i), expected_tensor_seq.Get(i), per_sample_tolerance, relative_per_sample_tolerance,
                                   post_processing);
      if (res.first != COMPARE_RESULT::SUCCESS) {
        return res;
      }
    }

    return std::make_pair(COMPARE_RESULT::SUCCESS, "");

  } else {
    // Maps
#if !defined(DISABLE_ML_OPS)
    if (o.Type() == DataTypeImpl::GetType<VectorMapInt64ToFloat>()) {
      return CompareSeqOfMapToFloat(o.Get<VectorMapInt64ToFloat>(), expected_mlvalue.Get<VectorMapInt64ToFloat>(),
                                    per_sample_tolerance, relative_per_sample_tolerance, post_processing);
    }
    if (o.Type() == DataTypeImpl::GetType<VectorMapStringToFloat>()) {
      return CompareSeqOfMapToFloat(o.Get<VectorMapStringToFloat>(), expected_mlvalue.Get<VectorMapStringToFloat>(),
                                    per_sample_tolerance, relative_per_sample_tolerance, post_processing);
    }
    return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "");
#else
    return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "Map type is not supported in this build.");
#endif
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
      oss << utils::GetTensorShapeFromTensorShapeProto(tensor_shape_proto);
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

std::pair<COMPARE_RESULT, std::string> CompareOrtValueNumerals(const OrtValue& real_mlvalue, const OrtValue& expected_mlvalue,
                                                               double per_sample_tolerance,
                                                               double relative_per_sample_tolerance) {
  if (real_mlvalue.IsTensor()) {
    const Tensor& outvalue = real_mlvalue.Get<Tensor>();
    const Tensor& expected_tensor = expected_mlvalue.Get<Tensor>();
    if (outvalue.DataType() == DataTypeImpl::GetType<float>() &&
        expected_tensor.DataType() == DataTypeImpl::GetType<MLFloat16>()) {
      return CompareFloat16WithFloatResult<float, MLFloat16>(outvalue,
                                                             expected_tensor,
                                                             per_sample_tolerance,
                                                             relative_per_sample_tolerance);
    } else if (outvalue.DataType() == DataTypeImpl::GetType<MLFloat16>() &&
               expected_tensor.DataType() == DataTypeImpl::GetType<float>()) {
      return CompareFloat16WithFloatResult<MLFloat16, float>(outvalue,
                                                             expected_tensor,
                                                             per_sample_tolerance,
                                                             relative_per_sample_tolerance);
    }
  }

  return std::make_pair(COMPARE_RESULT::NOT_SUPPORT, "Unsupported compare with CompareOrtValueNumerals.");
}

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
    MLDataType p = DataTypeImpl::TypeFromProto(v.type());
    MLDataType q = ((OrtValue*)(const OrtValue*)o)->Type();
    if (q != p) {
      return std::make_pair(COMPARE_RESULT::TYPE_MISMATCH, "");
    }
  }

  return std::make_pair(COMPARE_RESULT::SUCCESS, "");
}
}  // namespace onnxruntime
