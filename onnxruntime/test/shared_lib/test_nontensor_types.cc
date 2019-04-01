// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "test_fixture.h"
#include <functional>
#include <set>
#include "test_allocator.h"
#include <iostream>

using namespace onnxruntime;

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

TEST_F(CApiTest, CreateGetVectorOfMapsInt64Float) {  // support zipmap output type seq(map(int64, float))
  // Creation
  std::unique_ptr<MockedOrtAllocator> default_allocator(std::make_unique<MockedOrtAllocator>());
  OrtAllocatorInfo* info;
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &info));
  std::unique_ptr<OrtAllocatorInfo, decltype(&OrtReleaseAllocatorInfo)> rel_info(info, OrtReleaseAllocatorInfo);

  RelAllocations<OrtValue> rel(&OrtReleaseValue);
  RelAllocations<OrtStatus> rels(&OrtReleaseStatus);

  const int N = 3;
  const int NUM_KV_PAIRS = 4;
  std::vector<OrtValue*> in(N);
  std::vector<int64_t> keys{3, 1, 2, 0};
  std::vector<int64_t> dims = {4};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  for (int i = 0; i < N; ++i) {
    // create key tensor
    OrtValue* keys_tensor = OrtCreateTensorWithDataAsOrtValue(info, keys.data(), keys.size() * sizeof(int64_t),
                                                              dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ASSERT_NE(keys_tensor, nullptr);
    rel.add(keys_tensor);

    // create value tensor
    OrtValue* values_tensor = OrtCreateTensorWithDataAsOrtValue(info, values.data(), values.size() * sizeof(float),
                                                                dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_NE(values_tensor, nullptr);
    rel.add(values_tensor);

    // create map ort value
    std::vector<OrtValue*> map_in{keys_tensor, values_tensor};
    OrtValue* map_ort = nullptr;
    OrtStatus* stx = OrtCreateValue(map_in.data(), 2, ONNX_TYPE_MAP, &map_ort);
    rels.add(stx);
    rel.add(map_ort);
    ASSERT_EQ(stx, nullptr);
    ASSERT_NE(map_ort, nullptr);

    in[i] = map_ort;
  }

  // repeat above 3 steps N times and store the result in an OrtValue array
  // create sequence ort value
  OrtValue* seq_ort = nullptr;
  OrtStatus* sty = OrtCreateValue(in.data(), N, ONNX_TYPE_SEQUENCE, &seq_ort);
  rels.add(sty);
  rel.add(seq_ort);
  ASSERT_EQ(sty, nullptr);
  ASSERT_NE(seq_ort, nullptr);

  // Get count
  size_t num_values = 0;
  OrtStatus* st2 = OrtGetValueCount(seq_ort, &num_values);
  rels.add(st2);
  ASSERT_EQ(st2, nullptr);
  ASSERT_EQ(num_values, N);

  // test negative case
  OrtValue* tmp = nullptr;
  OrtStatus* st_temp = OrtGetValue(seq_ort, 999, default_allocator.get(), &tmp);
  rels.add(st_temp);
  rel.add(tmp);
  ASSERT_NE(st_temp, nullptr);

  // Fetch
  for (int idx = 0; idx < N; ++idx) {
    OrtValue* map_out = nullptr;
    OrtStatus* st = OrtGetValue(seq_ort, idx, default_allocator.get(), &map_out);
    rel.add(map_out);
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(map_out, nullptr);

    // fetch the map
    // first fetch the keys
    OrtValue* keys_ort = nullptr;
    st = OrtGetValue(map_out, 0, default_allocator.get(), &keys_ort);
    rel.add(keys_ort);
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(keys_ort, nullptr);

    std::unique_ptr<int64_t> keys_ret_u;
    int64_t* keys_ret = keys_ret_u.get();
    st = OrtGetTensorMutableData(keys_ort, reinterpret_cast<void**>(&keys_ret));
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(keys_ret, nullptr);
    ASSERT_EQ(std::set<int64_t>(keys_ret, keys_ret + NUM_KV_PAIRS),
              std::set<int64_t>(std::begin(keys), std::end(keys)));

    // second fetch the values
    OrtValue* values_ort = nullptr;
    st = OrtGetValue(map_out, 1, default_allocator.get(), &values_ort);
    rel.add(values_ort);
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(values_ort, nullptr);

    std::unique_ptr<float> values_ret_u;
    float* values_ret = values_ret_u.get();
    st = OrtGetTensorMutableData(values_ort, reinterpret_cast<void**>(&values_ret));
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(values_ret, nullptr);
    ASSERT_EQ(std::set<float>(values_ret, values_ret + NUM_KV_PAIRS),
              std::set<float>(std::begin(values), std::end(values)));
  }
}

TEST_F(CApiTest, CreateGetVectorOfMapsStringFloat) {  // support zipmap output type seq(map(string, float))
  // Creation
  std::unique_ptr<MockedOrtAllocator> default_allocator(std::make_unique<MockedOrtAllocator>());
  OrtAllocatorInfo* info;
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &info));
  std::unique_ptr<OrtAllocatorInfo, decltype(&OrtReleaseAllocatorInfo)> rel_info(info, OrtReleaseAllocatorInfo);

  RelAllocations<OrtValue> rel(&OrtReleaseValue);
  RelAllocations<OrtStatus> rels(&OrtReleaseStatus);

  const int N = 3;
  const int64_t NUM_KV_PAIRS = 4;
  std::vector<OrtValue*> in(N);
  const char* keys_arr[NUM_KV_PAIRS] = {"abc", "def", "ghi", "jkl"};
  std::vector<std::string> keys{keys_arr, keys_arr + NUM_KV_PAIRS};
  std::vector<int64_t> dims = {NUM_KV_PAIRS};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  for (int i = 0; i < N; ++i) {
    // create key tensor
    OrtValue* keys_tensor = OrtCreateTensorWithDataAsOrtValue(info, keys.data(), keys.size() * sizeof(std::string),
                                                              dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    ASSERT_NE(keys_tensor, nullptr);
    rel.add(keys_tensor);

    // create value tensor
    OrtValue* values_tensor = OrtCreateTensorWithDataAsOrtValue(info, values.data(), values.size() * sizeof(float),
                                                                dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_NE(values_tensor, nullptr);
    rel.add(values_tensor);

    // create map ort value
    std::vector<OrtValue*> map_in{keys_tensor, values_tensor};
    OrtValue* map_ort = nullptr;
    OrtStatus* stx = OrtCreateValue(map_in.data(), 2, ONNX_TYPE_MAP, &map_ort);
    rels.add(stx);
    rel.add(map_ort);
    ASSERT_EQ(stx, nullptr);
    ASSERT_NE(map_ort, nullptr);

    in[i] = map_ort;
  }

  // repeat above 3 steps N times and store the result in an OrtValue array
  // create sequence ort value
  OrtValue* seq_ort = nullptr;
  OrtStatus* sty = OrtCreateValue(in.data(), N, ONNX_TYPE_SEQUENCE, &seq_ort);
  rels.add(sty);
  rel.add(seq_ort);
  ASSERT_EQ(sty, nullptr);
  ASSERT_NE(seq_ort, nullptr);

  // Get count
  size_t num_values;
  OrtStatus* st2 = OrtGetValueCount(seq_ort, &num_values);
  rels.add(st2);
  ASSERT_EQ(st2, nullptr);
  ASSERT_EQ(num_values, N);

  // Fetch
  for (int idx = 0; idx < N; ++idx) {
    OrtValue* map_out = nullptr;
    OrtStatus* st = OrtGetValue(seq_ort, idx, default_allocator.get(), &map_out);
    rel.add(map_out);
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(map_out, nullptr);

    // fetch the map
    // first fetch the keys
    OrtValue* keys_ort = nullptr;
    st = OrtGetValue(map_out, 0, default_allocator.get(), &keys_ort);
    rel.add(keys_ort);
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(keys_ort, nullptr);

    size_t data_len;
    st = OrtGetStringTensorDataLength(keys_ort, &data_len);
    rels.add(st);
    ASSERT_EQ(st, nullptr);

    std::string result(data_len, '\0');
    std::vector<size_t> offsets(NUM_KV_PAIRS);
    st = OrtGetStringTensorContent(keys_ort, (void*)result.data(), data_len, offsets.data(), offsets.size());
    rels.add(st);
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
    OrtValue* values_ort = nullptr;
    st = OrtGetValue(map_out, 1, default_allocator.get(), &values_ort);
    rel.add(values_ort);
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(values_ort, nullptr);

    std::unique_ptr<float> values_ret_u;
    float* values_ret = values_ret_u.get();
    st = OrtGetTensorMutableData(values_ort, reinterpret_cast<void**>(&values_ret));
    rels.add(st);
    ASSERT_EQ(st, nullptr);
    ASSERT_NE(values_ret, nullptr);
    ASSERT_EQ(std::set<float>(values_ret, values_ret + NUM_KV_PAIRS),
              std::set<float>(std::begin(values), std::end(values)));
  }
}
