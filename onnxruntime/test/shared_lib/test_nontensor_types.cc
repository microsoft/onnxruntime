// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <iostream>
#include <set>

#include "core/common/common.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test_allocator.h"

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

  const size_t N = 3;
  const int NUM_KV_PAIRS = 4;
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

  // test negative case
  bool failed = false;
  ORT_TRY {
    auto temp = seq_ort.GetValue(999, default_allocator.get());
  }
  ORT_CATCH(const Ort::Exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      failed = e.GetOrtErrorCode() == ORT_RUNTIME_EXCEPTION;
    });
  }

  ASSERT_EQ(failed, true);

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

  const size_t N = 3;
  const int64_t NUM_KV_PAIRS = 4;
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

  const int64_t NUM_KV_PAIRS = 4;
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

  //It doesn't own the pointer -
  //The destructor of the "Unowned" struct will release the ownership (and thus prevent the pointer from being double freed)
  auto map_type_info = type_info.GetMapTypeInfo();

  //Check key type
  ASSERT_EQ(map_type_info.GetMapKeyType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

  //It owns the pointer
  Ort::TypeInfo map_value_type_info = map_type_info.GetMapValueType();

  //Check value type and shape
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
  const int N = 2;
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
  const int N = 2;
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
  const int N = 2;
  for (int i = 0; i < N; ++i) {
    // create tensor
    Ort::Value tensor = Ort::Value::CreateTensor(info, vals.data(), vals.size() * sizeof(int64_t),
                                                 dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    in.push_back(std::move(tensor));
  }

  Ort::Value seq_ort = Ort::Value::CreateSequence(in);
  Ort::TypeInfo type_info = seq_ort.GetTypeInfo();

  //It doesn't own the pointer -
  //The destructor of the "Unowned" struct will release the ownership (and thus prevent the pointer from being double freed)
  auto seq_type_info = type_info.GetSequenceTypeInfo();

  ASSERT_EQ(seq_type_info.GetSequenceElementType().GetONNXType(), ONNX_TYPE_TENSOR);
  // No shape present, as sequence allows different shapes for each element
  // ASSERT_EQ(seq_type_info.GetSequenceElementType().GetTensorTypeAndShapeInfo().GetShape(), dims);
  ASSERT_EQ(seq_type_info.GetSequenceElementType().GetTensorTypeAndShapeInfo().GetElementType(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}
