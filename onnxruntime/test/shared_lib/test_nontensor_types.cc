// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "test_fixture.h"
#include <functional>
#include <set>
#include "test_allocator.h"

using namespace onnxruntime;

template <typename T>
struct RelAllocations {
  RelAllocations(std::function<void(T*)> f) : relf(f) {}
  std::vector<T*> torel;
  std::function<void(T*)> relf;
  ~RelAllocations() {
    for (auto x : torel) {
      if (x)
        relf(x);
    }
  }
};

TEST_F(CApiTest, CreateGetVectorOfMapsInt64Float) {
  // Creation
  OrtAllocatorInfo* info;
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &info));
  std::unique_ptr<OrtAllocatorInfo, decltype(&OrtReleaseAllocatorInfo)> rel_info(info, OrtReleaseAllocatorInfo);

  RelAllocations<OrtValue> rel(&OrtReleaseValue);
  RelAllocations<OrtStatus> rel_status(&OrtReleaseStatus);

  const int N = 3;
  const int NUM_KV_PAIRS = 4;
  std::vector<OrtValue*> in(N);
  std::vector<int64_t> keys{3, 1, 2, 0};
  std::vector<size_t> dims = {4};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  for (int i = 0; i < N; ++i) {
    // create key tensor
    OrtValue* keys_tensor = OrtCreateTensorWithDataAsOrtValue(info, keys.data(), keys.size() * sizeof(int64_t), dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    rel.torel.push_back(keys_tensor);

    // create value tensor
    OrtValue* values_tensor = OrtCreateTensorWithDataAsOrtValue(info, values.data(), values.size() * sizeof(float), dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    rel.torel.push_back(values_tensor);

    // create map ort value
    std::vector<OrtValue*> map_in{keys_tensor, values_tensor};
    OrtValue* map_ort = nullptr;
    OrtStatus* stx = OrtCreateValue(map_in.data(), 2, ONNX_TYPE_MAP, info, &map_ort);
    rel_status.torel.push_back(stx);
    rel.torel.push_back(map_ort);
    ASSERT_EQ(stx, nullptr);

    in[i] = map_ort;
  }

  // repeat above 3 steps N times and store the result in an OrtValue array
  // create sequence ort value
  OrtValue* seq_ort = nullptr;
  OrtStatus* sty = OrtCreateValue(in.data(), N, ONNX_TYPE_SEQUENCE, info, &seq_ort);
  rel_status.torel.push_back(sty);
  rel.torel.push_back(seq_ort);
  ASSERT_EQ(sty, nullptr);

  // Get count
  int num_values;
  OrtStatus* st2 = OrtGetNumValues(seq_ort, &num_values);
  rel_status.torel.push_back(st2);
  ASSERT_EQ(st2, nullptr);
  ASSERT_EQ(num_values, N);

  // Fetch
  for (int idx = 0; idx < N; ++idx) {
    OrtValue* map_out = nullptr;
    OrtStatus* st = OrtGetValue(seq_ort, idx, info, &map_out);
    rel.torel.push_back(map_out);
    rel_status.torel.push_back(st);
    ASSERT_EQ(st, nullptr);

    // fetch the map
    // first fetch the keys
    OrtValue* keys_ort = nullptr;
    st = OrtGetValue(map_out, 0, info, &keys_ort);
    rel.torel.push_back(keys_ort);
    rel_status.torel.push_back(st);
    ASSERT_EQ(st, nullptr);

    int64_t* keys_ret = nullptr;
    st = OrtGetTensorMutableData(keys_ort, reinterpret_cast<void**>(&keys_ret));
    rel_status.torel.push_back(st);
    // TODO free keys_ret
    ASSERT_EQ(st, nullptr);
    ASSERT_EQ(std::set<int64_t>(keys_ret, keys_ret + NUM_KV_PAIRS), std::set<int64_t>(std::begin(keys), std::end(keys)));

    // second fetch the values
    OrtValue* values_ort = nullptr;
    st = OrtGetValue(map_out, 1, info, &values_ort);
    rel.torel.push_back(values_ort);
    rel_status.torel.push_back(st);
    ASSERT_EQ(st, nullptr);

    float* values_ret = nullptr;
    st = OrtGetTensorMutableData(values_ort, reinterpret_cast<void**>(&values_ret));
    rel_status.torel.push_back(st);
    // free values_ret
    ASSERT_EQ(st, nullptr);
    ASSERT_EQ(std::set<float>(values_ret, values_ret + NUM_KV_PAIRS), std::set<float>(std::begin(values), std::end(values)));
  }
}

TEST_F(CApiTest, CreateGetVectorOfMapsStringFloat) {
  // Creation
  OrtAllocatorInfo* info;
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &info));
  std::unique_ptr<OrtAllocatorInfo, decltype(&OrtReleaseAllocatorInfo)> rel_info(info, OrtReleaseAllocatorInfo);

  RelAllocations<OrtValue> rel(&OrtReleaseValue);
  RelAllocations<OrtStatus> rel_status(&OrtReleaseStatus);

  const int N = 3;
  const int NUM_KV_PAIRS = 4;
  std::vector<OrtValue*> in(N);
  const char* keys_arr[] = {"abc", "def", "ghi", "jkl"};
  std::vector<std::string> keys{keys_arr, keys_arr + NUM_KV_PAIRS};
  std::vector<size_t> dims = {4};
  std::vector<float> values{3.0f, 1.0f, 2.f, 0.f};
  for (int i = 0; i < N; ++i) {
    // create key tensor
    OrtValue* keys_tensor = nullptr;
    std::unique_ptr<MockedOrtAllocator> default_allocator(std::make_unique<MockedOrtAllocator>());
    OrtStatus* stx = ::OrtCreateTensorAsOrtValue(default_allocator.get(), dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &keys_tensor);
    rel_status.torel.push_back(stx);
    rel.torel.push_back(keys_tensor);
    stx = OrtFillStringTensor(keys_tensor, keys_arr, 4);
    rel_status.torel.push_back(stx);

    // create value tensor
    OrtValue* values_tensor = OrtCreateTensorWithDataAsOrtValue(info, values.data(), values.size() * sizeof(float), dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    rel.torel.push_back(values_tensor);

    // create map ort value
    std::vector<OrtValue*> map_in{keys_tensor, values_tensor};
    OrtValue* map_ort = nullptr;
    stx = OrtCreateValue(map_in.data(), 2, ONNX_TYPE_MAP, info, &map_ort);
    rel_status.torel.push_back(stx);
    rel.torel.push_back(map_ort);
    ASSERT_EQ(stx, nullptr);

    in[i] = map_ort;
  }

  // repeat above 3 steps N times and store the result in an OrtValue array
  // create sequence ort value
  OrtValue* seq_ort = nullptr;
  OrtStatus* sty = OrtCreateValue(in.data(), N, ONNX_TYPE_SEQUENCE, info, &seq_ort);
  rel_status.torel.push_back(sty);
  rel.torel.push_back(seq_ort);
  ASSERT_EQ(sty, nullptr);

  // Get count
  int num_values;
  OrtStatus* st2 = OrtGetNumValues(seq_ort, &num_values);
  rel_status.torel.push_back(st2);
  ASSERT_EQ(st2, nullptr);
  ASSERT_EQ(num_values, N);

  // Fetch
  for (int idx = 0; idx < N; ++idx) {
    OrtValue* map_out = nullptr;
    OrtStatus* st = OrtGetValue(seq_ort, idx, info, &map_out);
    rel.torel.push_back(map_out);
    rel_status.torel.push_back(st);
    ASSERT_EQ(st, nullptr);

    // fetch the map
    // first fetch the keys
    OrtValue* keys_ort = nullptr;
    st = OrtGetValue(map_out, 0, info, &keys_ort);
    rel.torel.push_back(keys_ort);
    rel_status.torel.push_back(st);
    ASSERT_EQ(st, nullptr);

    size_t data_len;
    st = OrtGetStringTensorDataLength(keys_ort, &data_len);
    rel_status.torel.push_back(st);
    std::string result(data_len, '\0');
    std::vector<size_t> offsets(NUM_KV_PAIRS);
    st = OrtGetStringTensorContent(keys_ort, (void*)result.data(), data_len, offsets.data(), offsets.size());
    rel_status.torel.push_back(st);

    const char* s = result.data();
    size_t* o = offsets.data();
    size_t end = *o;
    std::set<std::string> keys_ret;
    for (size_t i = 0, start = 0; i < data_len; ++i, ++o) {
      std::string stemp(s + start, s + end);
      std::cout << "s: " << stemp << std::endl;
      keys_ret.insert(stemp);
      start += end;
    }
    ASSERT_EQ(keys_ret, std::set<std::string>(std::begin(keys), std::end(keys)));

    // second fetch the values
    OrtValue* values_ort = nullptr;
    st = OrtGetValue(map_out, 1, info, &values_ort);
    rel.torel.push_back(values_ort);
    rel_status.torel.push_back(st);
    ASSERT_EQ(st, nullptr);

    float* values_ret = nullptr;
    st = OrtGetTensorMutableData(values_ort, reinterpret_cast<void**>(&values_ret));
    rel_status.torel.push_back(st);
    // TODO free values_ret
    ASSERT_EQ(st, nullptr);
    ASSERT_EQ(std::set<float>(values_ret, values_ret + NUM_KV_PAIRS), std::set<float>(std::begin(values), std::end(values)));
  }
}
