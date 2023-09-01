// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <iostream>
#include <set>

#include "core/common/common.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test_allocator.h"

#include "core/common/gsl.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

template <typename T>
struct RelAllocations {
  RelAllocations(std::function<void(T*)> f) : relf(f) {}
  void add(T* x) {
    torel.push_back(x);
  }
  std::vector<T*> torel;
  std::function<void(T*)> relf;
  ~RelAllocations() {
    for (auto x : torel) {
      if (x)
        relf(x);
    }
  }
};

#if !defined(DISABLE_ML_OPS)
TEST(CApiTest, CreateGetVectorOfMapsInt64Float) {  // support zipmap output type seq(map(int64, float))
  // Creation
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  constexpr size_t N = 3;
  constexpr int NUM_KV_PAIRS = 4;
  std::vector<Ort::Value> in;
  std::vector<int64_t> keys{3, 1, 2, 0};
  std::vector<int64_t> dims = {4};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  for (size_t i = 0; i < N; ++i) {
    // create key tensor
    Ort::Value keys_tensor = Ort::Value::CreateTensor(info, keys.data(), keys.size() * sizeof(int64_t),
                                                      dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    // create value tensor
    Ort::Value values_tensor = Ort::Value::CreateTensor(info, values.data(), values.size() * sizeof(float),
                                                        dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    // create map ort value
    in.emplace_back(Ort::Value::CreateMap(keys_tensor, values_tensor));
  }

  // repeat above 3 steps N times and store the result in an OrtValue array
  // create sequence ort value
  Ort::Value seq_ort = Ort::Value::CreateSequence(in);

  // Get count
  size_t num_values = seq_ort.GetCount();
  ASSERT_EQ(num_values, N);

#if !defined(ORT_NO_EXCEPTIONS)
  // test negative case
  bool failed = false;
  try {
    auto temp = seq_ort.GetValue(999, default_allocator.get());
  } catch (const Ort::Exception& e) {
    failed = e.GetOrtErrorCode() == ORT_RUNTIME_EXCEPTION;
  }
  ASSERT_EQ(failed, true);
#endif  // !defined(ORT_NO_EXCEPTIONS)

  // Fetch
  for (size_t idx = 0; idx < N; ++idx) {
    Ort::Value map_out = seq_ort.GetValue(static_cast<int>(idx), default_allocator.get());

    // fetch the map
    // first fetch the keys
    Ort::Value keys_ort = map_out.GetValue(0, default_allocator.get());

    int64_t* keys_ret = keys_ort.GetTensorMutableData<int64_t>();
    ASSERT_EQ(std::set<int64_t>(keys_ret, keys_ret + NUM_KV_PAIRS),
              std::set<int64_t>(std::begin(keys), std::end(keys)));

    // second fetch the values
    Ort::Value values_ort = map_out.GetValue(1, default_allocator.get());

    float* values_ret = values_ort.GetTensorMutableData<float>();
    ASSERT_EQ(std::set<float>(values_ret, values_ret + NUM_KV_PAIRS),
              std::set<float>(std::begin(values), std::end(values)));
  }
}

TEST(CApiTest, CreateGetVectorOfMapsStringFloat) {  // support zipmap output type seq(map(string, float))
  // Creation
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  constexpr size_t N = 3;
  constexpr int64_t NUM_KV_PAIRS = 4;
  std::vector<Ort::Value> in;
  const char* keys_arr[NUM_KV_PAIRS] = {"abc", "def", "ghi", "jkl"};
  std::vector<std::string> keys{keys_arr, keys_arr + NUM_KV_PAIRS};
  std::vector<int64_t> dims = {NUM_KV_PAIRS};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  for (size_t i = 0; i < N; ++i) {
    // create key tensor
    Ort::Value keys_tensor = Ort::Value::CreateTensor(info, keys.data(), keys.size() * sizeof(std::string),
                                                      dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    // create value tensor
    Ort::Value values_tensor = Ort::Value::CreateTensor(info, values.data(), values.size() * sizeof(float),
                                                        dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // create map ort value
    in.emplace_back(Ort::Value::CreateMap(keys_tensor, values_tensor));
  }

  // repeat above 3 steps N times and store the result in an OrtValue array
  // create sequence ort value
  Ort::Value seq_ort = Ort::Value::CreateSequence(in);

  // Get count
  size_t num_values = seq_ort.GetCount();
  ASSERT_EQ(num_values, N);

  // Fetch
  for (size_t idx = 0; idx < N; ++idx) {
    Ort::Value map_out = seq_ort.GetValue(static_cast<int>(idx), default_allocator.get());

    // fetch the map
    // first fetch the keys
    Ort::Value keys_ort = map_out.GetValue(0, default_allocator.get());

    size_t data_len = keys_ort.GetStringTensorDataLength();

    std::string result(data_len, '\0');
    std::vector<size_t> offsets(NUM_KV_PAIRS);
    keys_ort.GetStringTensorContent((void*)result.data(), data_len, offsets.data(), offsets.size());

    const char* s = result.data();
    std::set<std::string> keys_ret;
    for (size_t i = 0; i < offsets.size(); ++i) {
      size_t start = offsets[i];
      size_t count = (i + 1) < offsets.size() ? offsets[i + 1] - start : data_len - start;
      std::string stemp(s + start, count);
      keys_ret.insert(stemp);
    }
    ASSERT_EQ(keys_ret, std::set<std::string>(std::begin(keys), std::end(keys)));

    // second fetch the values
    Ort::Value values_ort = map_out.GetValue(1, default_allocator.get());

    float* values_ret = values_ort.GetTensorMutableData<float>();
    ASSERT_EQ(std::set<float>(values_ret, values_ret + NUM_KV_PAIRS),
              std::set<float>(std::begin(values), std::end(values)));
  }
}
#endif  // !defined(DISABLE_ML_OPS)

TEST(CApiTest, TypeInfoMap) {
  // Creation
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  constexpr int64_t NUM_KV_PAIRS = 4;
  std::vector<int64_t> keys{0, 1, 2, 3};
  std::vector<int64_t> dims = {NUM_KV_PAIRS};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  // create key tensor
  Ort::Value keys_tensor = Ort::Value::CreateTensor(info, keys.data(), keys.size() * sizeof(int64_t),
                                                    dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  // create value tensor
  Ort::Value values_tensor = Ort::Value::CreateTensor(info, values.data(), values.size() * sizeof(float),
                                                      dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

#if !defined(DISABLE_ML_OPS)
  Ort::Value map_ort = Ort::Value::CreateMap(keys_tensor, values_tensor);
  Ort::TypeInfo type_info = map_ort.GetTypeInfo();

  // It doesn't own the pointer -
  // The destructor of the "Unowned" struct will release the ownership (and thus prevent the pointer from being double freed)
  auto map_type_info = type_info.GetMapTypeInfo();

  // Check key type
  ASSERT_EQ(map_type_info.GetMapKeyType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  // It owns the pointer
  Ort::TypeInfo map_value_type_info = map_type_info.GetMapValueType();

  // Check value type and shape
  ASSERT_EQ(map_value_type_info.GetONNXType(), ONNX_TYPE_TENSOR);
  // No shape present, as map values allow different shapes for each element
  // ASSERT_EQ(map_value_type_info.GetTensorTypeAndShapeInfo().GetShape(), dims);
  ASSERT_EQ(map_value_type_info.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

#else

#if !defined(ORT_NO_EXCEPTIONS)
  // until https://github.com/google/googletest/pull/2904/ makes it into a release,
  // check an exception is thrown with the expected message the ugly way
  try {
    Ort::Value map_ort = Ort::Value::CreateMap(keys_tensor, values_tensor);
    ASSERT_TRUE(false) << "CreateMap should have throw in this build";
  } catch (const Ort::Exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Map type is not supported in this build"));
  }
#endif
#endif
}

TEST(CApiTest, CreateGetSeqTensors) {
  // Creation
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> in;
  std::vector<int64_t> vals{3, 1, 2, 0};
  std::vector<int64_t> dims{1, 4};
  constexpr int N = 2;
  for (int i = 0; i < N; ++i) {
    // create tensor
    Ort::Value tensor = Ort::Value::CreateTensor(info, vals.data(), vals.size() * sizeof(int64_t),
                                                 dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    in.push_back(std::move(tensor));
  }

  Ort::Value seq_ort = Ort::Value::CreateSequence(in);

  // Fetch
  for (int idx = 0; idx < N; ++idx) {
    Ort::Value out = seq_ort.GetValue(idx, default_allocator.get());
    int64_t* ret = out.GetTensorMutableData<int64_t>();
    ASSERT_EQ(std::set<int64_t>(ret, ret + vals.size()),
              std::set<int64_t>(std::begin(vals), std::end(vals)));
  }
}

TEST(CApiTest, CreateGetSeqStringTensors) {
  // Creation
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> in;
  const char* string_input_data[] = {"abs", "def"};
  constexpr int N = 2;
  for (int i = 0; i < N; ++i) {
    // create tensor
    std::vector<int64_t> shape{2};
    auto value = Ort::Value::CreateTensor(Ort::AllocatorWithDefaultOptions(), shape.data(), shape.size(),
                                          ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

    Ort::ThrowOnError(Ort::GetApi().FillStringTensor(value, string_input_data, 2));
    in.push_back(std::move(value));
  }

  Ort::Value seq_ort = Ort::Value::CreateSequence(in);

  // Fetch
  std::set<std::string> string_set;
  for (int idx = 0; idx < N; ++idx) {
    Ort::Value out = seq_ort.GetValue(idx, default_allocator.get());
    size_t data_len = out.GetStringTensorDataLength();
    std::string result(data_len, '\0');
    std::vector<size_t> offsets(N);
    out.GetStringTensorContent((void*)result.data(), data_len, offsets.data(), offsets.size());

    const char* s = result.data();
    for (size_t i = 0; i < offsets.size(); ++i) {
      size_t start = offsets[i];
      size_t count = (i + 1) < offsets.size() ? offsets[i + 1] - start : data_len - start;
      std::string stemp(s + start, count);
      string_set.insert(stemp);
    }
  }
  ASSERT_EQ(string_set, std::set<std::string>(std::begin(string_input_data), std::end(string_input_data)));
}

TEST(CApiTest, TypeInfoSequence) {
  // Creation
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> in;
  std::vector<int64_t> vals{3, 1, 2, 0};
  std::vector<int64_t> dims{1, 4};
  constexpr int N = 2;
  for (int i = 0; i < N; ++i) {
    // create tensor
    Ort::Value tensor = Ort::Value::CreateTensor(info, vals.data(), vals.size() * sizeof(int64_t),
                                                 dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    in.push_back(std::move(tensor));
  }

  Ort::Value seq_ort = Ort::Value::CreateSequence(in);
  Ort::TypeInfo type_info = seq_ort.GetTypeInfo();

  // It doesn't own the pointer -
  // The destructor of the "Unowned" struct will release the ownership (and thus prevent the pointer from being double freed)
  auto seq_type_info = type_info.GetSequenceTypeInfo();

  ASSERT_EQ(seq_type_info.GetSequenceElementType().GetONNXType(), ONNX_TYPE_TENSOR);
  // No shape present, as sequence allows different shapes for each element
  // ASSERT_EQ(seq_type_info.GetSequenceElementType().GetTensorTypeAndShapeInfo().GetShape(), dims);
  ASSERT_EQ(seq_type_info.GetSequenceElementType().GetTensorTypeAndShapeInfo().GetElementType(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

TEST(CPPApi, ConvertFloatToFloat16) {
  // Test data type
  {
    constexpr float sample = 1.0f;
    Ort::Float16_t flt16(sample);
    EXPECT_FALSE(flt16.IsNaN());
    auto int_rep = flt16.val;
    const Ort::Float16_t flt_from_int = Ort::Float16_t::FromBits(int_rep);
    EXPECT_FALSE(flt_from_int.IsNaN());
    EXPECT_EQ(flt16, flt_from_int);
    const double diff = std::fabs(sample - flt_from_int.ToFloat());
    if (diff > FLT_EPSILON || (std::isnan(diff) && !std::isnan(sample))) {
      EXPECT_TRUE(false);
    }
  }
  // Test bulk conversion
  {
    const float sample[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<Ort::Float16_t> converted;
    converted.reserve(std::size(sample));

    // Run conversion
    std::transform(std::begin(sample), std::end(sample), std::back_inserter(converted),
                   [](float v) { return Ort::Float16_t(v); });

    for (size_t i = 0; i < std::size(sample); ++i) {
      EXPECT_FALSE(converted[i].IsNaN());
      const double diff = std::fabs(sample[i] - converted[i].ToFloat());
      if ((std::isnan(diff) && !std::isnan(sample[i])) || diff > FLT_EPSILON) {
        EXPECT_TRUE(false);
      }
    }

    std::vector<float> back_converted;
    back_converted.reserve(std::size(sample));
    std::transform(converted.cbegin(), converted.cend(), std::back_inserter(back_converted),
                   [](const Ort::Float16_t& bf) { return static_cast<float>(bf); });

    for (size_t i = 0; i < std::size(sample); ++i) {
      EXPECT_FALSE(std::isnan(back_converted[i]));
      const double diff = std::fabs(sample[i] - back_converted[i]);
      if ((std::isnan(diff) && !std::isnan(sample[i])) || diff > FLT_EPSILON) {
        EXPECT_TRUE(false);
      }
    }
  }
}

TEST(CPPApi, Float16Zeros) {
  const auto positive_zero = Ort::Float16_t::FromBits(0x0000);
  EXPECT_FALSE(positive_zero.IsNegative());
  const float float_positive_zero = static_cast<float>(positive_zero);
  EXPECT_EQ(+0.0f, float_positive_zero);
  EXPECT_FALSE(std::signbit(float_positive_zero));

  const auto negative_zero = Ort::Float16_t::FromBits(0x8000);
  EXPECT_TRUE(negative_zero.IsNegative());
  const float float_positive_negzero = static_cast<float>(negative_zero);
  EXPECT_EQ(-0.0f, float_positive_negzero);
  EXPECT_TRUE(std::signbit(float_positive_negzero));
}

namespace {
const auto EpsilonFl16 = Ort::Float16_t::FromBits(Ort::Float16_t::kEpsilonBits);
const auto NaNFl16 = Ort::Float16_t::FromBits(Ort::Float16_t::kPositiveQNaNBits);
const auto MinValueFl16 = Ort::Float16_t::FromBits(Ort::Float16_t::kMinValueBits);
const auto MaxValueFl16 = Ort::Float16_t::FromBits(Ort::Float16_t::kMaxValueBits);
const auto InfinityFl16 = Ort::Float16_t::FromBits(Ort::Float16_t::kPositiveInfinityBits);
}  // namespace

TEST(CPPApi, Float16Comparision) {
  const auto left = Ort::Float16_t(-33.33f);
  const auto left_same = Ort::Float16_t(-33.33f);
  const auto right = Ort::Float16_t(66.66f);
  const auto right_same = Ort::Float16_t(66.66f);

  EXPECT_LT(EpsilonFl16, right);

  EXPECT_EQ(left, left_same);
  EXPECT_NE(left, left_same.Negate());

  EXPECT_EQ(right, right_same);
  EXPECT_NE(right, right_same.Negate());

  EXPECT_LT(left, right);
  EXPECT_LT(right.Negate(), left);
  EXPECT_LT(left.Negate(), right);
}

TEST(CPPApi, Float16TestNAN) {
  const Ort::Float16_t fp16NANFromSingle(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(fp16NANFromSingle.IsNaN());

  // NaN are not equal to each other
  EXPECT_NE(NaNFl16, fp16NANFromSingle);

  const float NanFromBFloat16 = fp16NANFromSingle.ToFloat();
  EXPECT_TRUE(std::isnan(NanFromBFloat16));

  EXPECT_FALSE(MaxValueFl16.IsNaN());
}

TEST(CPPApi, Float16NaNComparision) {
  EXPECT_FALSE(NaNFl16 < NaNFl16);
  EXPECT_TRUE(NaNFl16 != NaNFl16);
  EXPECT_FALSE(NaNFl16 == NaNFl16);

  EXPECT_FALSE(MaxValueFl16 < NaNFl16);
  EXPECT_FALSE(MaxValueFl16 == NaNFl16);
  EXPECT_FALSE(NaNFl16 < MinValueFl16);

  EXPECT_LT(MinValueFl16, MaxValueFl16);
}

TEST(CPPApi, Float16Infinity) {
  EXPECT_FALSE(MinValueFl16.IsInfinity());
  EXPECT_FALSE(MaxValueFl16.IsInfinity());
  EXPECT_TRUE(MaxValueFl16.IsFinite());

  const Ort::Float16_t pos_infinity_from_float(std::numeric_limits<float>::infinity());
  EXPECT_TRUE(pos_infinity_from_float.IsInfinity());
  EXPECT_FALSE(pos_infinity_from_float.IsFinite());
  EXPECT_FALSE(pos_infinity_from_float.IsNegative());

  const Ort::Float16_t neg_infinity_from_float(-std::numeric_limits<float>::infinity());
  EXPECT_TRUE(neg_infinity_from_float.IsInfinity());
  EXPECT_FALSE(neg_infinity_from_float.IsFinite());
  EXPECT_TRUE(neg_infinity_from_float.IsNegative());

  const float pos_infinity_from_bfloat16 = static_cast<float>(InfinityFl16);
  EXPECT_TRUE(std::isinf(pos_infinity_from_bfloat16));
}

TEST(CPPApi, Float16NormalSubnormal) {
  EXPECT_FALSE(InfinityFl16.IsNormal());
  EXPECT_TRUE(Ort::Float16_t(45.6f).IsNormal());
  EXPECT_FALSE(Ort::Float16_t(45.6f).IsSubnormal());

  // 0b0_0000_0000_000_0001 ~0.000000059604645
  constexpr uint16_t min_subnormal_bits = 0x0001;
  const Ort::Float16_t smallest_subnormal = Ort::Float16_t::FromBits(min_subnormal_bits);
  EXPECT_TRUE(smallest_subnormal.IsSubnormal());
  EXPECT_FALSE(smallest_subnormal.IsNormal());

  // float smallest positive subnormal is ~1.40129846432481707092E-45, and
  // in float the same number above would be normal
  const float float_from_smallest_subnormal = static_cast<float>(smallest_subnormal);
  EXPECT_TRUE(std::isnormal(float_from_smallest_subnormal));

  // 0b0_0000_0000_111_1111; ~0.000060975552
  constexpr uint16_t max_subnormal_bits = 0x007F;
  const Ort::Float16_t largest_subnormal = Ort::Float16_t::FromBits(max_subnormal_bits);
  EXPECT_TRUE(largest_subnormal.IsSubnormal());
  EXPECT_FALSE(largest_subnormal.IsNormal());

  // However, in float the same number above would be normal
  const float float_from_largest_subnormal = static_cast<float>(largest_subnormal);
  EXPECT_TRUE(std::isnormal(float_from_largest_subnormal));
}

TEST(CPPApi, BFloat16ConvertFloatToBFloat16) {
  // Test data type
  {
    constexpr float sample = 1.0f;
    Ort::BFloat16_t flt16(sample);
    EXPECT_FALSE(flt16.IsNaN());
    auto int_rep = flt16.val;
    const Ort::BFloat16_t flt_from_int = Ort::BFloat16_t::FromBits(int_rep);
    EXPECT_FALSE(flt_from_int.IsNaN());
    EXPECT_EQ(flt16, flt_from_int);
    const double diff = std::fabs(sample - flt_from_int.ToFloat());
    if (diff > FLT_EPSILON || (std::isnan(diff) && !std::isnan(sample))) {
      EXPECT_TRUE(false);
    }
  }
  // Test bulk conversion
  {
    const float sample[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<Ort::BFloat16_t> converted;
    converted.reserve(std::size(sample));

    // Run conversion
    std::transform(std::begin(sample), std::end(sample), std::back_inserter(converted),
                   [](float v) { return Ort::BFloat16_t(v); });

    for (size_t i = 0; i < std::size(sample); ++i) {
      EXPECT_FALSE(converted[i].IsNaN());
      const double diff = std::fabs(sample[i] - converted[i].ToFloat());
      if ((std::isnan(diff) && !std::isnan(sample[i])) || diff > FLT_EPSILON) {
        EXPECT_TRUE(false);
      }
    }

    std::vector<float> back_converted;
    back_converted.reserve(std::size(sample));
    std::transform(converted.cbegin(), converted.cend(), std::back_inserter(back_converted),
                   [](const Ort::BFloat16_t& bf) { return static_cast<float>(bf); });

    for (size_t i = 0; i < std::size(sample); ++i) {
      EXPECT_FALSE(std::isnan(back_converted[i]));
      const double diff = std::fabs(sample[i] - back_converted[i]);
      if ((std::isnan(diff) && !std::isnan(sample[i])) || diff > FLT_EPSILON) {
        EXPECT_TRUE(false);
      }
    }
  }
}

TEST(CPPApi, BFloat16Zeros) {
  const auto positive_zero = Ort::BFloat16_t::FromBits(0x0000);
  EXPECT_FALSE(positive_zero.IsNegative());
  const float float_positive_zero = static_cast<float>(positive_zero);
  EXPECT_EQ(+0.0f, float_positive_zero);
  EXPECT_FALSE(std::signbit(float_positive_zero));

  const auto negative_zero = Ort::BFloat16_t::FromBits(0x8000);
  EXPECT_TRUE(negative_zero.IsNegative());
  const float float_positive_negzero = static_cast<float>(negative_zero);
  EXPECT_EQ(-0.0f, float_positive_negzero);
  EXPECT_TRUE(std::signbit(float_positive_negzero));
}

namespace {
const auto EpsilonBfl16 = Ort::BFloat16_t::FromBits(Ort::BFloat16_t::kEpsilonBits);
const auto NaNBfl15 = Ort::BFloat16_t::FromBits(Ort::BFloat16_t::kPositiveQNaNBits);
const auto MinValueBfl16 = Ort::BFloat16_t::FromBits(Ort::BFloat16_t::kMinValueBits);
const auto MaxValueBfl16 = Ort::BFloat16_t::FromBits(Ort::BFloat16_t::kMaxValueBits);
const auto InfinityBFl16 = Ort::BFloat16_t::FromBits(Ort::BFloat16_t::kPositiveInfinityBits);
}  // namespace

TEST(CPPApi, BFloat16Comparision) {
  const auto left = Ort::BFloat16_t(-33.33f);
  const auto left_same = Ort::BFloat16_t(-33.33f);
  const auto right = Ort::BFloat16_t(66.66f);
  const auto right_same = Ort::BFloat16_t(66.66f);

  EXPECT_LT(EpsilonBfl16, right);

  EXPECT_EQ(left, left_same);
  EXPECT_NE(left, left_same.Negate());

  EXPECT_EQ(right, right_same);
  EXPECT_NE(right, right_same.Negate());

  EXPECT_LT(left, right);
  EXPECT_LT(right.Negate(), left);
  EXPECT_LT(left.Negate(), right);
}

TEST(CPPApi, BFloat16TestNAN) {
  const Ort::BFloat16_t fp16NANFromSingle(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(fp16NANFromSingle.IsNaN());

  // NaN are not equal to each other
  EXPECT_NE(NaNBfl15, fp16NANFromSingle);

  const float NanFromBFloat16 = fp16NANFromSingle.ToFloat();
  EXPECT_TRUE(std::isnan(NanFromBFloat16));

  EXPECT_FALSE(MaxValueBfl16.IsNaN());
}

TEST(CPPApi, BFloat16NaNComparision) {
  EXPECT_FALSE(NaNBfl15 < NaNBfl15);
  EXPECT_TRUE(NaNBfl15 != NaNBfl15);
  EXPECT_FALSE(NaNBfl15 == NaNBfl15);

  EXPECT_FALSE(MaxValueBfl16 < NaNBfl15);
  EXPECT_FALSE(MaxValueBfl16 == NaNBfl15);
  EXPECT_FALSE(NaNBfl15 < MinValueBfl16);

  EXPECT_LT(MinValueBfl16, MaxValueBfl16);
}

TEST(CPPApi, BFloat16Infinity) {
  EXPECT_FALSE(MinValueBfl16.IsInfinity());
  EXPECT_FALSE(MaxValueBfl16.IsInfinity());
  EXPECT_TRUE(MaxValueBfl16.IsFinite());

  const Ort::BFloat16_t pos_infinity_from_float(std::numeric_limits<float>::infinity());
  EXPECT_TRUE(pos_infinity_from_float.IsInfinity());
  EXPECT_FALSE(pos_infinity_from_float.IsFinite());
  EXPECT_FALSE(pos_infinity_from_float.IsNegative());

  const Ort::BFloat16_t neg_infinity_from_float(-std::numeric_limits<float>::infinity());
  EXPECT_TRUE(neg_infinity_from_float.IsInfinity());
  EXPECT_FALSE(neg_infinity_from_float.IsFinite());
  EXPECT_TRUE(neg_infinity_from_float.IsNegative());

  const float pos_infinity_from_bfloat16 = static_cast<float>(InfinityBFl16);
  EXPECT_TRUE(std::isinf(pos_infinity_from_bfloat16));
}

TEST(CPPApi, BFloat16NormalSubnormal) {
  EXPECT_FALSE(InfinityBFl16.IsNormal());
  EXPECT_TRUE(Ort::BFloat16_t(45.6f).IsNormal());
  EXPECT_FALSE(Ort::BFloat16_t(45.6f).IsSubnormal());

  // 0b0_0000_0000_000_0001
  constexpr uint16_t min_subnormal_bits = 0x0001;
  const Ort::BFloat16_t smallest_subnormal = Ort::BFloat16_t::FromBits(min_subnormal_bits);
  EXPECT_TRUE(smallest_subnormal.IsSubnormal());
  EXPECT_FALSE(smallest_subnormal.IsNormal());

  const float float_from_smallest_subnormal = static_cast<float>(smallest_subnormal);
  EXPECT_FALSE(std::isnormal(float_from_smallest_subnormal));

  // 0b0_0000_0000_111_1111;
  constexpr uint16_t max_subnormal_bits = 0x007F;
  const Ort::BFloat16_t largest_subnormal = Ort::BFloat16_t::FromBits(max_subnormal_bits);
  EXPECT_TRUE(largest_subnormal.IsSubnormal());
  EXPECT_FALSE(largest_subnormal.IsNormal());

  const float float_from_largest_subnormal = static_cast<float>(largest_subnormal);
  EXPECT_FALSE(std::isnormal(float_from_largest_subnormal));
}

#if !defined(DISABLE_SPARSE_TENSORS)
TEST(CApiTest, SparseTensorUsingAPI) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  {
    // COO
    const std::vector<int64_t> dense_shape{3, 3};
    const std::vector<int64_t> values_shape{3};
    std::vector<int32_t> expected_values = {1, 1, 1};
    constexpr int64_t values_len = 3;
    std::vector<int64_t> expected_linear_indices = {2, 3, 5};
    const std::vector<int64_t> indices_shape{3};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    Ort::Value::Shape ort_values_shape{&values_len, 1U};
    auto coo_st = Ort::Value::CreateSparseTensor(info, expected_values.data(), ort_dense_shape, ort_values_shape);
    coo_st.UseCooIndices(expected_linear_indices.data(), expected_linear_indices.size());

    {
      auto ti = coo_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }

    {
      auto t_type_shape = coo_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }

    ASSERT_EQ(ORT_SPARSE_COO, coo_st.GetSparseFormat());

    {
      auto values_ts = coo_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());
    }

    {
      const auto* values = coo_st.GetSparseTensorValues<int32_t>();
      auto val_span = gsl::make_span(values, gsl::narrow_cast<size_t>(values_shape[0]));
      ASSERT_TRUE(std::equal(expected_values.cbegin(), expected_values.cend(), val_span.begin(), val_span.end()));
    }

    {
      auto indices_ts = coo_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_COO_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(indices_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = coo_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_COO_INDICES, num_indices);
      ASSERT_EQ(num_indices, static_cast<size_t>(indices_shape[0]));
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_linear_indices.cbegin(), expected_linear_indices.cend(), ind_span.begin(), ind_span.end()));
    }
  }

  {
    // CSR test
    const std::vector<int64_t> dense_shape{3, 3};
    const std::vector<int64_t> values_shape{3};
    const std::vector<int64_t> inner_shape{3};
    const std::vector<int64_t> outer_shape{4};
    std::vector<int32_t> expected_values = {1, 1, 1};
    const std::vector<std::string> expected_values_str = {"1", "1", "1"};
    std::vector<int64_t> expected_inner = {2, 0, 2};
    std::vector<int64_t> expected_outer = {0, 1, 3, 3};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    constexpr int64_t values_len = 3;
    Ort::Value::Shape ort_values_shape{&values_len, 1U};
    auto csr_st = Ort::Value::CreateSparseTensor(info, expected_values.data(), ort_dense_shape, ort_values_shape);
    csr_st.UseCsrIndices(expected_inner.data(), expected_inner.size(), expected_outer.data(), expected_outer.size());
    {
      auto ti = csr_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }

    {
      auto t_type_shape = csr_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }

    ASSERT_EQ(ORT_SPARSE_CSRC, csr_st.GetSparseFormat());

    {
      auto values_ts = csr_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());
    }

    {
      const auto* values = csr_st.GetSparseTensorValues<int32_t>();
      auto val_span = gsl::make_span(values, expected_values.size());
      ASSERT_TRUE(std::equal(expected_values.cbegin(), expected_values.cend(), val_span.begin(), val_span.end()));
    }

    {
      auto indices_ts = csr_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_CSR_INNER_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(inner_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = csr_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_CSR_INNER_INDICES, num_indices);
      ASSERT_EQ(num_indices, expected_inner.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_inner.cbegin(), expected_inner.cend(), ind_span.begin(), ind_span.end()));
    }

    {
      auto indices_ts = csr_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_CSR_OUTER_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(outer_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = csr_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_CSR_OUTER_INDICES, num_indices);
      ASSERT_EQ(num_indices, expected_outer.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_outer.cbegin(), expected_outer.cend(), ind_span.begin(), ind_span.end()));
    }
  }

  {
    // BlockSparse test
    const std::vector<int64_t> dense_shape{8, 8};
    constexpr int64_t block_size = 2;
    const std::vector<int64_t> values_shape{2, block_size, block_size};
    // Two dense blocks
    std::vector<int32_t> data_blocks{
        1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int64_t> indices_shape{2, 2};  // two blocks by two coordinates
    // (0, 0), (0,1)
    std::vector<int32_t> blocksparse_indices = {
        0, 0, 0, 1};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    Ort::Value::Shape ort_values_shape{values_shape.data(), values_shape.size()};
    auto bsp_st = Ort::Value::CreateSparseTensor(info, data_blocks.data(), ort_dense_shape, ort_values_shape);
    bsp_st.UseBlockSparseIndices({indices_shape.data(), indices_shape.size()}, blocksparse_indices.data());
    {
      auto ti = bsp_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }
    {
      auto t_type_shape = bsp_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }
    ASSERT_EQ(ORT_SPARSE_BLOCK_SPARSE, bsp_st.GetSparseFormat());
    {
      auto values_ts = bsp_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());
    }
    {
      const auto* values = bsp_st.GetSparseTensorValues<int32_t>();
      auto val_span = gsl::make_span(values, data_blocks.size());
      ASSERT_TRUE(std::equal(data_blocks.cbegin(), data_blocks.cend(), val_span.begin(), val_span.end()));
    }
    {
      auto indices_ts = bsp_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_BLOCK_SPARSE_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, indices_ts.GetElementType());
      ASSERT_EQ(indices_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int32_t* indices = bsp_st.GetSparseTensorIndicesData<int32_t>(ORT_SPARSE_BLOCK_SPARSE_INDICES, num_indices);
      ASSERT_EQ(num_indices, blocksparse_indices.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(blocksparse_indices.cbegin(), blocksparse_indices.cend(), ind_span.begin(), ind_span.end()));
    }
  }
}

TEST(CApiTest, SparseTensorFillSparseTensorFormatAPI) {
  auto allocator = Ort::AllocatorWithDefaultOptions();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  {
    // COO
    const std::vector<int64_t> dense_shape{3, 3};
    const std::vector<int64_t> values_shape{3};
    std::vector<int32_t> expected_values = {1, 1, 1};
    constexpr int64_t values_len = 3;
    std::vector<int64_t> expected_linear_indices = {2, 3, 5};
    const std::vector<int64_t> indices_shape{3};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    auto coo_st = Ort::Value::CreateSparseTensor<int32_t>(allocator, ort_dense_shape);
    coo_st.FillSparseTensorCoo(info, {&values_len, 1U, {expected_values.data()}},
                               expected_linear_indices.data(), expected_linear_indices.size());
    {
      auto ti = coo_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }

    {
      auto t_type_shape = coo_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }

    ASSERT_EQ(ORT_SPARSE_COO, coo_st.GetSparseFormat());

    {
      auto values_ts = coo_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());
    }

    {
      const auto* values = coo_st.GetSparseTensorValues<int32_t>();
      auto val_span = gsl::make_span(values, gsl::narrow_cast<size_t>(values_shape[0]));
      ASSERT_TRUE(std::equal(expected_values.cbegin(), expected_values.cend(), val_span.begin(), val_span.end()));
    }

    {
      auto indices_ts = coo_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_COO_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(indices_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = coo_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_COO_INDICES, num_indices);
      ASSERT_EQ(num_indices, static_cast<size_t>(indices_shape[0]));
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_linear_indices.cbegin(), expected_linear_indices.cend(), ind_span.begin(), ind_span.end()));
    }
  }
  {
    // CSR test
    const std::vector<int64_t> dense_shape{3, 3};
    const std::vector<int64_t> values_shape{3};
    const std::vector<int64_t> inner_shape{3};
    const std::vector<int64_t> outer_shape{4};
    const std::vector<int32_t> expected_values = {1, 1, 1};
    const std::vector<int64_t> expected_inner = {2, 0, 2};
    const std::vector<int64_t> expected_outer = {0, 1, 3, 3};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    auto csr_st = Ort::Value::CreateSparseTensor<int32_t>(allocator, ort_dense_shape);
    csr_st.FillSparseTensorCsr(info, {values_shape.data(), values_shape.size(), {expected_values.data()}},
                               expected_inner.data(), expected_inner.size(),
                               expected_outer.data(), expected_outer.size());
    {
      auto ti = csr_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }

    {
      auto t_type_shape = csr_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }

    ASSERT_EQ(ORT_SPARSE_CSRC, csr_st.GetSparseFormat());

    {
      auto values_ts = csr_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());
    }

    {
      const auto* values = csr_st.GetSparseTensorValues<int32_t>();
      auto val_span = gsl::make_span(values, expected_values.size());
      ASSERT_TRUE(std::equal(expected_values.cbegin(), expected_values.cend(), val_span.begin(), val_span.end()));
    }

    {
      auto indices_ts = csr_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_CSR_INNER_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(inner_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = csr_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_CSR_INNER_INDICES, num_indices);
      ASSERT_EQ(num_indices, expected_inner.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_inner.cbegin(), expected_inner.cend(), ind_span.begin(), ind_span.end()));
    }

    {
      auto indices_ts = csr_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_CSR_OUTER_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(outer_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = csr_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_CSR_OUTER_INDICES, num_indices);
      ASSERT_EQ(num_indices, expected_outer.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_outer.cbegin(), expected_outer.cend(), ind_span.begin(), ind_span.end()));
    }
  }
  {
    // BlockSparse test
    const std::vector<int64_t> dense_shape{8, 8};
    constexpr int64_t block_size = 2;
    const std::vector<int64_t> values_shape{2, block_size, block_size};
    // Two dense blocks
    std::vector<int32_t> data_blocks{
        1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int64_t> indices_shape{2, 2};  // two blocks by two coordinates
    // (0, 0), (0,1)
    std::vector<int32_t> blocksparse_indices = {
        0, 0, 0, 1};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    auto bsp_st = Ort::Value::CreateSparseTensor<int32_t>(allocator, ort_dense_shape);
    bsp_st.FillSparseTensorBlockSparse(info, {values_shape.data(), values_shape.size(), {data_blocks.data()}},
                                       {indices_shape.data(), indices_shape.size()}, blocksparse_indices.data());
    {
      auto ti = bsp_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }
    {
      auto t_type_shape = bsp_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }
    ASSERT_EQ(ORT_SPARSE_BLOCK_SPARSE, bsp_st.GetSparseFormat());
    {
      auto values_ts = bsp_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());
    }
    {
      const auto* values = bsp_st.GetSparseTensorValues<int32_t>();
      auto val_span = gsl::make_span(values, data_blocks.size());
      ASSERT_TRUE(std::equal(data_blocks.cbegin(), data_blocks.cend(), val_span.begin(), val_span.end()));
    }
    {
      auto indices_ts = bsp_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_BLOCK_SPARSE_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, indices_ts.GetElementType());
      ASSERT_EQ(indices_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int32_t* indices = bsp_st.GetSparseTensorIndicesData<int32_t>(ORT_SPARSE_BLOCK_SPARSE_INDICES, num_indices);
      ASSERT_EQ(num_indices, blocksparse_indices.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(blocksparse_indices.cbegin(), blocksparse_indices.cend(), ind_span.begin(), ind_span.end()));
    }
  }
}

TEST(CApiTest, SparseTensorFillSparseFormatStringsAPI) {
  auto allocator = Ort::AllocatorWithDefaultOptions();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  {
    // COO
    const std::vector<int64_t> dense_shape{3, 3};
    const std::vector<int64_t> values_shape{3};
    std::vector<std::string> expected_values = {"1", "1", "1"};
    const char* const strings[] = {"1", "1", "1"};
    constexpr int64_t values_len = 3;
    std::vector<int64_t> expected_linear_indices = {2, 3, 5};
    const std::vector<int64_t> indices_shape{3};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    auto coo_st = Ort::Value::CreateSparseTensor(allocator, ort_dense_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    coo_st.FillSparseTensorCoo(info, {&values_len, 1U, {strings}},
                               expected_linear_indices.data(), expected_linear_indices.size());
    {
      auto ti = coo_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }

    {
      auto t_type_shape = coo_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }

    ASSERT_EQ(ORT_SPARSE_COO, coo_st.GetSparseFormat());

    {
      auto values_ts = coo_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());

      for (size_t i = 0; i < values_len; ++i) {
        const auto& ex = expected_values[i];
        size_t len = coo_st.GetStringTensorElementLength(i);
        ASSERT_EQ(ex.size(), len);
        auto buffer = std::make_unique<char[]>(len);
        coo_st.GetStringTensorElement(len, i, buffer.get());
        ASSERT_EQ(0, ex.compare(0U, ex.size(), buffer.get(), len));
      }

      size_t data_len = coo_st.GetStringTensorDataLength();
      auto buffer = std::make_unique<char[]>(data_len);
      auto offsets = std::make_unique<size_t[]>(expected_values.size());
      /// XXX: Do something about this API.
      /// Need to add N + 1 terminating offset, or skip the first zero offset
      /// altogether and add the N + 1
      coo_st.GetStringTensorContent(buffer.get(), data_len, offsets.get(), values_len);
      for (size_t i = 0, limit = expected_values.size(); i < limit; ++i) {
        const auto& ex = expected_values[i];
        const char* p = &buffer[offsets[i]];
        size_t len = (i == (limit - 1)) ? (data_len - offsets[i]) : offsets[i + 1] - offsets[i];
        ASSERT_EQ(ex.size(), len);
        std::string s(p, len);
        ASSERT_EQ(expected_values[i], s);
      }
    }

    {
      auto indices_ts = coo_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_COO_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(indices_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = coo_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_COO_INDICES, num_indices);
      ASSERT_EQ(num_indices, static_cast<size_t>(indices_shape[0]));
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_linear_indices.cbegin(), expected_linear_indices.cend(), ind_span.begin(), ind_span.end()));
    }
  }
  {
    // CSR strings
    const std::vector<int64_t> dense_shape{3, 3};
    const std::vector<int64_t> values_shape{3};
    const std::vector<int64_t> inner_shape{3};
    const std::vector<int64_t> outer_shape{4};
    const std::vector<std::string> expected_values{"1", "1", "1"};
    const char* const strings[] = {"1", "1", "1"};
    const std::vector<int64_t> expected_inner{2, 0, 2};
    const std::vector<int64_t> expected_outer{0, 1, 3, 3};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    const int64_t values_len = static_cast<int64_t>(expected_values.size());
    auto csr_st = Ort::Value::CreateSparseTensor(allocator, ort_dense_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    csr_st.FillSparseTensorCsr(info, {values_shape.data(), values_shape.size(), {strings}},
                               expected_inner.data(), expected_inner.size(),
                               expected_outer.data(), expected_outer.size());
    {
      auto ti = csr_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }

    {
      auto t_type_shape = csr_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }

    ASSERT_EQ(ORT_SPARSE_CSRC, csr_st.GetSparseFormat());
    {
      auto values_ts = csr_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());

      for (size_t i = 0; i < static_cast<size_t>(values_len); ++i) {
        const auto& ex = expected_values[i];
        size_t len = csr_st.GetStringTensorElementLength(i);
        ASSERT_EQ(ex.size(), len);
        auto buffer = std::make_unique<char[]>(len);
        csr_st.GetStringTensorElement(len, i, buffer.get());
        ASSERT_EQ(0, ex.compare(0U, ex.size(), buffer.get(), len));
      }

      size_t data_len = csr_st.GetStringTensorDataLength();
      auto buffer = std::make_unique<char[]>(data_len);
      auto offsets = std::make_unique<size_t[]>(expected_values.size());
      /// XXX: Do something about this API.
      /// Need to add N + 1 terminating offset, or skip the first zero offset
      /// altogether and add the N + 1
      csr_st.GetStringTensorContent(buffer.get(), data_len, offsets.get(), gsl::narrow_cast<size_t>(values_len));
      for (size_t i = 0, limit = expected_values.size(); i < limit; ++i) {
        const auto& ex = expected_values[i];
        const char* p = &buffer[offsets[i]];
        size_t len = (i == (limit - 1)) ? (data_len - offsets[i]) : offsets[i + 1] - offsets[i];
        ASSERT_EQ(ex.size(), len);
        std::string s(p, len);
        ASSERT_EQ(ex, s);
      }
    }
    {
      auto indices_ts = csr_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_CSR_INNER_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(inner_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = csr_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_CSR_INNER_INDICES, num_indices);
      ASSERT_EQ(num_indices, expected_inner.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_inner.cbegin(), expected_inner.cend(), ind_span.begin(), ind_span.end()));
    }

    {
      auto indices_ts = csr_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_CSR_OUTER_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
      ASSERT_EQ(outer_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int64_t* indices = csr_st.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_CSR_OUTER_INDICES, num_indices);
      ASSERT_EQ(num_indices, expected_outer.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(expected_outer.cbegin(), expected_outer.cend(), ind_span.begin(), ind_span.end()));
    }
  }
  {
    // BlockSparse test
    const std::vector<int64_t> dense_shape{8, 8};
    constexpr int64_t block_size = 2;
    const std::vector<int64_t> values_shape{2, block_size, block_size};
    // Two dense blocks
    const std::vector<std::string> data_blocks{
        "1", "2", "3", "4", "5", "6", "7", "8"};
    const char* const strings[] = {"1", "2", "3", "4", "5", "6", "7", "8"};
    const std::vector<int64_t> indices_shape{2, 2};  // two blocks by two coordinates
    // (0, 0), (0,1)
    std::vector<int32_t> blocksparse_indices = {
        0, 0, 0, 1};

    Ort::Value::Shape ort_dense_shape{dense_shape.data(), dense_shape.size()};
    auto bsp_st = Ort::Value::CreateSparseTensor(allocator, ort_dense_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    bsp_st.FillSparseTensorBlockSparse(info, {values_shape.data(), values_shape.size(), {strings}},
                                       {indices_shape.data(), indices_shape.size()}, blocksparse_indices.data());
    {
      auto ti = bsp_st.GetTypeInfo();
      ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
      auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, tensor_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), tensor_type_shape.GetDimensionsCount());
    }
    {
      auto t_type_shape = bsp_st.GetTensorTypeAndShapeInfo();
      ASSERT_EQ(dense_shape, t_type_shape.GetShape());
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, t_type_shape.GetElementType());
      ASSERT_EQ(dense_shape.size(), t_type_shape.GetDimensionsCount());
    }
    ASSERT_EQ(ORT_SPARSE_BLOCK_SPARSE, bsp_st.GetSparseFormat());
    {
      auto values_ts = bsp_st.GetSparseTensorValuesTypeAndShapeInfo();
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, values_ts.GetElementType());
      ASSERT_EQ(values_shape, values_ts.GetShape());

      for (size_t i = 0, limit = data_blocks.size(); i < limit; ++i) {
        const auto& ex = data_blocks[i];
        size_t len = bsp_st.GetStringTensorElementLength(i);
        ASSERT_EQ(ex.size(), len);
        auto buffer = std::make_unique<char[]>(len);
        bsp_st.GetStringTensorElement(len, i, buffer.get());
        ASSERT_EQ(0, ex.compare(0U, ex.size(), buffer.get(), len));
      }

      size_t data_len = bsp_st.GetStringTensorDataLength();
      auto buffer = std::make_unique<char[]>(data_len);
      /// XXX: Do something about this API.
      /// Need to add N + 1 terminating offset, or skip the first zero offset
      /// altogether and add the N + 1
      auto offsets = std::make_unique<size_t[]>(data_blocks.size());
      bsp_st.GetStringTensorContent(buffer.get(), data_len, offsets.get(), data_blocks.size());
      for (size_t i = 0, limit = data_blocks.size(); i < limit; ++i) {
        const auto& ex = data_blocks[i];
        const char* p = &buffer[offsets[i]];
        size_t len = (i == (limit - 1)) ? (data_len - offsets[i]) : offsets[i + 1] - offsets[i];
        ASSERT_EQ(ex.size(), len);
        std::string s(p, len);
        ASSERT_EQ(ex, s);
      }
    }
    {
      auto indices_ts = bsp_st.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_BLOCK_SPARSE_INDICES);
      ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, indices_ts.GetElementType());
      ASSERT_EQ(indices_shape, indices_ts.GetShape());

      size_t num_indices = 0;
      const int32_t* indices = bsp_st.GetSparseTensorIndicesData<int32_t>(ORT_SPARSE_BLOCK_SPARSE_INDICES, num_indices);
      ASSERT_EQ(num_indices, blocksparse_indices.size());
      auto ind_span = gsl::make_span(indices, num_indices);
      ASSERT_TRUE(std::equal(blocksparse_indices.cbegin(), blocksparse_indices.cend(), ind_span.begin(), ind_span.end()));
    }
  }
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)
