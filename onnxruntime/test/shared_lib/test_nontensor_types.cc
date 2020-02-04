// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/make_unique.h>
#include "core/session/onnxruntime_cxx_api.h"
#include <functional>
#include <set>
#include "test_allocator.h"
#include <gtest/gtest.h>
#include <iostream>

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

TEST(CApiTest, CreateGetVectorOfMapsInt64Float) {  // support zipmap output type seq(map(int64, float))
  // Creation
  auto default_allocator = onnxruntime::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  const size_t N = 3;
  const int NUM_KV_PAIRS = 4;
  std::vector<Ort::Value> in;
  std::vector<int64_t> keys{3, 1, 2, 0};
  std::vector<int64_t> dims = {4};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  for (int i = 0; i < N; ++i) {
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
  try {
    auto temp = seq_ort.GetValue(999, default_allocator.get());
  } catch (const Ort::Exception& e) {
    failed = e.GetOrtErrorCode() == ORT_RUNTIME_EXCEPTION;
  }
  ASSERT_EQ(failed, true);

  // Fetch
  for (int idx = 0; idx < N; ++idx) {
    Ort::Value map_out = seq_ort.GetValue(idx, default_allocator.get());

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
  auto default_allocator = onnxruntime::make_unique<MockedOrtAllocator>();
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  const size_t N = 3;
  const int64_t NUM_KV_PAIRS = 4;
  std::vector<Ort::Value> in;
  const char* keys_arr[NUM_KV_PAIRS] = {"abc", "def", "ghi", "jkl"};
  std::vector<std::string> keys{keys_arr, keys_arr + NUM_KV_PAIRS};
  std::vector<int64_t> dims = {NUM_KV_PAIRS};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  for (int i = 0; i < N; ++i) {
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
  for (int idx = 0; idx < N; ++idx) {
    Ort::Value map_out = seq_ort.GetValue(idx, default_allocator.get());

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

TEST(CApiTest, CreateGetSeqTensors) {
  // Creation
  auto default_allocator = onnxruntime::make_unique<MockedOrtAllocator>();
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
