// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/checkers.h"

#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/framework/TensorSeq.h"

#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
struct DefaultTolerance;

template <>
struct DefaultTolerance<double> {
  static constexpr float absolute = 1e-5f;
  static constexpr float relative = 1e-5f;

  // Allow to have different default absolute tolerance for different providers.
  static float get_absolute(const std::string& /*provider_type*/) {
    return absolute;
  }
};

template <>
struct DefaultTolerance<float> {
#if defined(ENABLE_TRAINING)
  static constexpr float absolute = 1e-3f;
#else
  static constexpr float absolute = 1e-5f;
#endif

  static constexpr float relative = 1e-4f;

  static float get_absolute(const std::string& /*provider_type*/) {
    return absolute;
  }
};

template <>
struct DefaultTolerance<MLFloat16> {
#if defined(ENABLE_TRAINING)
  static constexpr float absolute = 0.005f;
#else
  // The thresholds for inference are estimated with PyTorch script like the following:
  //    x = torch.rand(1000, 1000)
  //    absolute = ((x + 1e-6).to(torch.float16) - x).abs().max() * 10
  //    x[abs(x) < absolute] = absolute
  //    relative = ((x - x.to(torch.float16)) / x).abs().max() * 2
  static constexpr float absolute = 0.0025f;
#endif

  static constexpr float relative = 0.001f;

  static float get_absolute(const std::string& provider_type) {
    if (provider_type == kDmlExecutionProvider) {
      return 0.005f;
    }
    return absolute;
  }
};

template <>
struct DefaultTolerance<BFloat16> {
  // The thresholds for inference are estimated with PyTorch script like the following:
  //    x = torch.rand(1000, 1000)
  //    absolute = ((x + 1e-6).to(torch.bfloat16) - x).abs().max() * 10
  //    x[abs(x) < absolute] = absolute
  //    relative = ((x - x.to(torch.bfloat16)) / x).abs().max() * 2
  static constexpr float absolute = 0.02f;
  static constexpr float relative = 0.01f;

  static float get_absolute(const std::string& /*provider_type*/) {
    return absolute;
  }
};

struct ToleranceParams {
  float absolute;
  float relative;
};

template <typename T>
ToleranceParams get_tolerance_params(const ValidateOutputParams& params, const std::string& provider_type) {
  ToleranceParams new_params;
  new_params.absolute = params.absolute_error.has_value() ? *(params.absolute_error) : DefaultTolerance<T>::get_absolute(provider_type);
  new_params.relative = params.relative_error.has_value() ? *(params.relative_error) : DefaultTolerance<T>::relative;
  return new_params;
}

template <typename T>
T get_tolerance(const ToleranceParams& params, T expected_value) {
  static_assert(std::is_floating_point<T>::value, "T must be a floating point type");

  // The formula is similar to numpy.isclose: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
  return static_cast<T>(params.absolute) + static_cast<T>(params.relative) * std::abs(expected_value);
}

template <typename T>
Tensor copy_sort(const Tensor& src, const AllocatorPtr& allocator) {
  Tensor result(src.DataType(), src.Shape(), allocator);
  memcpy(result.MutableDataRaw(), src.DataRaw(), src.SizeInBytes());
  auto dst_span = gsl::make_span(result.MutableData<T>(), result.MutableData<T>() + result.Shape().Size());
  std::sort(dst_span.begin(), dst_span.end());
  return result;
}

// Check functions for tensor types
template <typename T>
void sort_expected_and_actual_buffers(const Tensor& expected, Tensor& expected_sorted,
                                      const Tensor& actual, Tensor& actual_sorted) {
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  expected_sorted = copy_sort<T>(expected, allocator);
  actual_sorted = copy_sort<T>(actual, allocator);
}

// Check functions for tensor types
template <typename T>
void sort_expected_and_actual_buffers(std::vector<T>& expected,
                                      std::vector<T>& actual) {
  ORT_ENFORCE(expected.size() == actual.size(),
              "The 2 containers contain different number of elements");
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
}

// The default implementation compares for equality, specialized versions for
// other types are below
template <typename T>
struct TensorCheck {
  void operator()(const Tensor& expected, const Tensor& actual, const ValidateOutputParams& params,
                  const std::string& /*provider_type*/) const {
    Tensor expected_sorted, actual_sorted;
    const T* cur_expected;
    const T* cur_actual;
    const auto size = actual.Shape().Size();
    if (params.sort_output) {
      // if order can be jumbled in the output of an operator, sort both the
      // expected and output buffers prior to
      // comparison this is a "best-effort" algo and should satisfy the
      // requirement for the few ops that do require this
      // support without investing in a more sophisticated infrastructure for the
      // same
      sort_expected_and_actual_buffers<T>(expected, expected_sorted, actual, actual_sorted);
      cur_expected = expected_sorted.Data<T>();
      cur_actual = actual_sorted.Data<T>();
    } else {
      cur_expected = expected.Data<T>();
      cur_actual = actual.Data<T>();
    }

    for (int64_t i = 0; i < size; ++i) {
      EXPECT_EQ(cur_expected[i], cur_actual[i]) << "i:" << i;
    }
  }
};

template <>
struct TensorCheck<uint8_t> {
  void operator()(const Tensor& expected,
                  const Tensor& actual,
                  const ValidateOutputParams& params,
                  const std::string& provider_type) const {
    const bool has_abs_err = params.absolute_error.has_value();
    const bool has_rel_err = params.relative_error.has_value();

    Tensor expected_sorted, actual_sorted;
    const uint8_t* cur_expected;
    const uint8_t* cur_actual;
    const auto size = actual.Shape().Size();
    if (params.sort_output) {
      // if order can be jumbled in the output of an operator, sort both the
      // expected and output buffers prior to
      // comparison this is a "best-effort" algo and should satisfy the
      // requirement for the few ops that do require this
      // support without investing in a more sophisticated infrastructure for the
      // same
      sort_expected_and_actual_buffers<uint8_t>(expected, expected_sorted, actual, actual_sorted);
      cur_expected = expected_sorted.Data<uint8_t>();
      cur_actual = actual_sorted.Data<uint8_t>();
    } else {
      cur_expected = expected.Data<uint8_t>();
      cur_actual = actual.Data<uint8_t>();
    }

    // For uint8_t results, we only allow NNAPI/XNNPACK EP to have an error tolerance, see below for the reason
    // XNNPACK EP will always round to larger. For example, 0.1 will be rounded to 1.0
    // For any other EPs, we still expect an exact match for the results
    // TODO: Verify if DML can possibly have a ROUNDING_MODE parameter and conform to the other EPs #41968513
    if ((provider_type == kNnapiExecutionProvider || provider_type == kDmlExecutionProvider ||
         provider_type == kXnnpackExecutionProvider) &&
        (has_abs_err || has_rel_err)) {
      double threshold = has_abs_err ? *(params.absolute_error)
                                     : 0.0;

      for (int64_t i = 0; i < size; ++i) {
        if (has_rel_err) {
          EXPECT_NEAR(cur_expected[i], cur_actual[i],
                      *(params.relative_error) * cur_expected[i])  // expected[i] is unsigned, can't be negative
              << "i:" << i;
        } else {  // has_abs_err
          EXPECT_NEAR(cur_expected[i], cur_actual[i], threshold) << "i:" << i;
        }
      }
    } else {
      for (int64_t i = 0; i < size; ++i) {
        EXPECT_EQ(cur_expected[i], cur_actual[i]) << "i:" << i;
      }
    }
  }
};

template <>
struct TensorCheck<int8_t> {
  void operator()(const Tensor& expected,
                  const Tensor& actual,
                  const ValidateOutputParams& params,
                  const std::string& /*provider_type*/) const {
    Tensor expected_sorted, actual_sorted;
    const int8_t* cur_expected;
    const int8_t* cur_actual;
    const auto size = actual.Shape().Size();
    if (params.sort_output) {
      // if order can be jumbled in the output of an operator, sort both the
      // expected and output buffers prior to
      // comparison this is a "best-effort" algo and should satisfy the
      // requirement for the few ops that do require this
      // support without investing in a more sophisticated infrastructure for the
      // same
      sort_expected_and_actual_buffers<int8_t>(expected, expected_sorted, actual, actual_sorted);
      cur_expected = expected_sorted.Data<int8_t>();
      cur_actual = actual_sorted.Data<int8_t>();
    } else {
      cur_expected = expected.template Data<int8_t>();
      cur_actual = actual.template Data<int8_t>();
    }

    // When absolute error is less than 1 for int8, it has same effect as no tolerance.
    const bool has_abs_err = params.absolute_error.has_value() && *(params.absolute_error) >= 1.0f;

    // TODO: the relative error is not used for int8 yet.
    if (has_abs_err) {
      double threshold = *(params.absolute_error);

      for (int64_t i = 0; i < size; ++i) {
        EXPECT_NEAR(cur_expected[i], cur_actual[i], threshold) << "i:" << i;
      }
    } else {
      for (int64_t i = 0; i < size; ++i) {
        EXPECT_EQ(cur_expected[i], cur_actual[i]) << "i:" << i;
      }
    }
  }
};

template <>
struct TensorCheck<double> {
  void operator()(const Tensor& expected,
                  const Tensor& actual,
                  const ValidateOutputParams& params,
                  const std::string& provider_type) const {
    auto size = actual.Shape().Size();

    // deal with rare cases in which order of output data from a kernel MAY be
    // undefined
    Tensor expected_sorted, actual_sorted;
    const double* cur_expected;
    const double* cur_actual;
    if (params.sort_output) {
      sort_expected_and_actual_buffers<double>(expected, expected_sorted, actual, actual_sorted);
      cur_expected = expected_sorted.Data<double>();
      cur_actual = actual_sorted.Data<double>();
    } else {
      cur_expected = expected.Data<double>();
      cur_actual = actual.Data<double>();
    }

    auto tolerance_params = get_tolerance_params<double>(params, provider_type);

    for (int64_t i = 0; i < size; ++i) {
      // NOTE: Check isnan first to work around MSVC linker bug when /LTCG:incremental is specified.
      // If the isinf check is first the isnan check and branch gets omitted
      if (std::isnan(cur_expected[i])) {
        EXPECT_TRUE(std::isnan(cur_actual[i])) << "Expected NaN. i:" << i;
      } else if (std::isinf(cur_expected[i])) {  // Test infinity for equality
        EXPECT_EQ(cur_expected[i], cur_actual[i]) << "Expected infinity. i:" << i;
      } else {
        double tolerance = get_tolerance<double>(tolerance_params, cur_expected[i]);
        EXPECT_NEAR(cur_expected[i], cur_actual[i], tolerance) << "i:" << i;
      }
    }
  }
};

template <typename T>
void InternalNumericalCheck(const Tensor& expected,
                            const Tensor& actual,
                            const ValidateOutputParams& params,
                            const std::string& provider_type) {
  // deal with rare cases in which order of output data from a kernel MAY be
  // undefined
  Tensor expected_sorted, actual_sorted;
  const T* cur_expected;
  const T* cur_actual;
  auto size = actual.Shape().Size();
  if (params.sort_output) {
    sort_expected_and_actual_buffers<T>(expected, expected_sorted, actual, actual_sorted);
    cur_expected = expected_sorted.Data<T>();
    cur_actual = actual_sorted.Data<T>();
  } else {
    cur_expected = expected.Data<T>();
    cur_actual = actual.Data<T>();
  }

  auto tolerance_params = get_tolerance_params<T>(params, provider_type);

  for (int64_t i = 0; i < size; ++i) {
    // NOTE: Check isnan first to work around MSVC linker bug when /LTCG:incremental is specified.
    // If the isinf check is first the isnan check and branch gets omitted
    if (std::isnan(cur_expected[i])) {
      EXPECT_TRUE(std::isnan(cur_actual[i])) << "Expected NaN. i:" << i;
    } else if (std::isinf(cur_expected[i])) {  // Test infinity for equality
      EXPECT_EQ(cur_expected[i], cur_actual[i]) << "Expected infinity. i:" << i;
    } else {
      T tolerance = get_tolerance<T>(tolerance_params, cur_expected[i]);
      EXPECT_NEAR(cur_expected[i], cur_actual[i], tolerance) << "i:" << i;
    }
  }
}

template <>
struct TensorCheck<float> {
  void operator()(const Tensor& expected,
                  const Tensor& actual,
                  const ValidateOutputParams& params,
                  const std::string& provider_type) const {
    InternalNumericalCheck<float>(expected, actual, params, provider_type);
  }
};

template <>
struct TensorCheck<MLFloat16> {
  void operator()(const Tensor& expected,
                  const Tensor& actual,
                  const ValidateOutputParams& params,
                  const std::string& provider_type) const {
    auto* cur_expected = expected.Data<MLFloat16>();
    auto* cur_actual = actual.Data<MLFloat16>();
    auto size = actual.Shape().Size();

    std::vector<float> f_expected(size);
    std::vector<float> f_actual(size);
    ConvertMLFloat16ToFloat(cur_expected, f_expected.data(), static_cast<int>(size));
    ConvertMLFloat16ToFloat(cur_actual, f_actual.data(), static_cast<int>(size));

    // deal with rare cases in which order of output data from a kernel MAY be
    // undefined
    if (params.sort_output) {
      sort_expected_and_actual_buffers<float>(f_expected, f_actual);
    }

    auto tolerance_params = get_tolerance_params<MLFloat16>(params, provider_type);

    for (int64_t i = 0; i < size; ++i) {
      if (std::isnan(f_expected[i])) {
        EXPECT_TRUE(std::isnan(f_expected[i])) << "Expected NaN. i:" << i;
      } else if (std::isinf(f_expected[i])) {  // Test infinity for equality
        EXPECT_EQ(f_expected[i], f_actual[i]) << "Expected infinity. i:" << i;
      } else {
        float tolerance = get_tolerance<float>(tolerance_params, f_expected[i]);
        EXPECT_NEAR(f_expected[i], f_actual[i], tolerance) << "i:" << i;
      }
    }
  }
};

template <>
struct TensorCheck<BFloat16> {
  void operator()(const Tensor& expected,
                  const Tensor& actual,
                  const ValidateOutputParams& params,
                  const std::string& provider_type) const {
    auto* cur_expected = expected.Data<BFloat16>();
    auto* cur_actual = actual.Data<BFloat16>();
    auto size = actual.Shape().Size();

    std::vector<float> f_expected(size);
    std::vector<float> f_actual(size);
    BFloat16ToFloat(cur_expected, f_expected.data(), static_cast<size_t>(size));
    BFloat16ToFloat(cur_actual, f_actual.data(), static_cast<size_t>(size));

    // deal with rare cases in which order of output data from a kernel MAY be
    // undefined
    if (params.sort_output) {
      sort_expected_and_actual_buffers<float>(f_expected, f_actual);
    }

    auto tolerance_params = get_tolerance_params<BFloat16>(params, provider_type);

    for (int64_t i = 0; i < size; ++i) {
      if (std::isnan(f_expected[i])) {
        EXPECT_TRUE(std::isnan(f_expected[i])) << "Expected NaN. i:" << i;
      } else if (std::isinf(f_expected[i])) {  // Test infinity for equality
        EXPECT_EQ(f_expected[i], f_actual[i]) << "Expected infinity. i:" << i;
      } else {
        float tolerance = get_tolerance<float>(tolerance_params, f_expected[i]);
        EXPECT_NEAR(f_expected[i], f_actual[i], tolerance) << "i:" << i;
      }
    }
  }
};
}  // namespace

// default Check
template <typename T>
void Check(std::string_view name, const OrtValue& expected, const T& actual,
           const ValidateOutputParams& /*params*/, const std::string& /*provider_type*/) {
  EXPECT_EQ(expected.Get<T>(), actual) << "name: " << name;
}

// Check for Tensors
template <>
void Check<Tensor>(std::string_view name, const OrtValue& expected, const Tensor& actual,
                   const ValidateOutputParams& params, const std::string& provider_type) {
  const Tensor& expected_tensor = expected.Get<Tensor>();
  ORT_ENFORCE(expected_tensor.Shape() == actual.Shape(),
              "Expected output shape [", expected_tensor.Shape(),
              "] did not match run output shape [", actual.Shape(),
              "] for ", name);

  utils::MLTypeCallDispatcher<bool, float, double, uint8_t, uint16_t, uint32_t, uint64_t,
                              int8_t, int16_t, int32_t, int64_t, std::string,
#if !defined(DISABLE_FLOAT8_TYPES)

                              Float8E4M3FN, Float8E4M3FNUZ, Float8E5M2, Float8E5M2FNUZ,
#endif
                              MLFloat16, BFloat16>
      t_disp(actual.GetElementType());

  t_disp.Invoke<TensorCheck>(expected_tensor, actual, params, provider_type);
}

// Check for sequence of tensors
template <>
void Check<TensorSeq>(std::string_view name, const OrtValue& expected, const TensorSeq& actual,
                      const ValidateOutputParams& params, const std::string& provider_type) {
  const auto& exp_seq = expected.Get<TensorSeq>();

  // first ensure data types match
  EXPECT_EQ(exp_seq.DataType(), actual.DataType())
      << "Data types don't match for " << name << ". Expected : " << DataTypeImpl::ToString(exp_seq.DataType())
      << " Output: " << actual.DataType();

  // check num of contained tensors
  size_t expected_num_tensors = exp_seq.Size();
  size_t actual_num_tensors = actual.Size();
  EXPECT_EQ(expected_num_tensors, actual_num_tensors)
      << "Mismatch in number of tensors in the sequence for " << name
      << ". Expected: " << expected_num_tensors
      << " Output: " << actual_num_tensors;

  // now check the contents of the tensors
  auto element_type = exp_seq.DataType()->AsPrimitiveDataType()->GetDataType();
  utils::MLTypeCallDispatcher<bool, float, double, uint8_t, uint16_t, uint32_t, uint64_t,
                              int8_t, int16_t, int32_t, int64_t, std::string,
#if !defined(DISABLE_FLOAT8_TYPES)

                              Float8E4M3FN, Float8E4M3FNUZ, Float8E5M2, Float8E5M2FNUZ,
#endif
                              MLFloat16, BFloat16>
      t_disp(element_type);

  for (size_t i = 0; i < actual_num_tensors; ++i) {
    t_disp.Invoke<TensorCheck>(exp_seq.Get(i), actual.Get(i), params, provider_type);
  }
}

template <typename Type>
void CheckDispatch(MLDataType type, std::string_view name, const OrtValue& expected, const OrtValue& actual,
                   const ValidateOutputParams& params, const std::string& provider_type) {
  if (type == DataTypeImpl::GetType<Type>()) {
    Check<Type>(name, expected, actual.Get<Type>(), params, provider_type);
  } else {
    ORT_THROW("OpTester:Check() not implemented for output tensor type of ", type);
  }
}

template <typename Type, typename Next, typename... Types>
void CheckDispatch(MLDataType type, std::string_view name, const OrtValue& expected, const OrtValue& actual,
                   const ValidateOutputParams& params, const std::string& provider_type) {
  if (type == DataTypeImpl::GetType<Type>()) {
    Check<Type>(name, expected, actual.Get<Type>(), params, provider_type);
  } else {
    CheckDispatch<Next, Types...>(type, name, expected, actual, params, provider_type);
  }
}

void CheckOrtValuesAreEqual(std::string_view name, const OrtValue& expected, const OrtValue& actual,
                            const ValidateOutputParams& params, const std::string& provider_type) {
  // Include provider_type in any error output
  SCOPED_TRACE(MakeString("provider type: ", provider_type));

  CheckDispatch<
      Tensor,
#if !defined(DISABLE_ML_OPS)
      VectorMapStringToFloat, VectorMapInt64ToFloat,
#endif
      TensorSeq>(expected.Type(), name, expected, actual, params, provider_type);
}

}  // namespace test
}  // namespace onnxruntime
