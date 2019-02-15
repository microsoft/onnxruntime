// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "test_fixture.h"
#include <functional>

using namespace onnxruntime;

template <typename T>
struct RelAllocations {
  RelAllocations(std::function<void(T*)> f): relf(f) {}
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
  RelAllocations<OrtStatus> rels(&OrtReleaseStatus);

  const int N = 3;
  OrtValue* in[N];
  for (int i=0; i<N; ++i) {
    // create key tensor
    int64_t keys[] = {3, 1, 2, 0};
    constexpr size_t keys_length = sizeof(keys) / sizeof(keys[0]);
    std::vector<size_t> dims = {4};
    OrtValue* keys_tensor = OrtCreateTensorWithDataAsOrtValue(info, keys, keys_length * sizeof(int64_t), dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    rel.torel.push_back(keys_tensor);

    // create value tensor
    float values[] = {3.0f, 1.0f, 2.f, 0.f};
    constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
    OrtValue* values_tensor = OrtCreateTensorWithDataAsOrtValue(info, values, values_length * sizeof(float), dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    rel.torel.push_back(values_tensor);

    // create map ort value
    OrtValue* map_in[2] = {keys_tensor, values_tensor};
    OrtValue* map_ort = nullptr;
    OrtStatus* st = OrtCreateValue(map_in, 2, ONNX_TYPE_MAP, info, &map_ort);
    rels.torel.push_back(st);
    ASSERT_EQ(OrtGetErrorCode(st), ORT_OK);

    in[i] = map_ort;
  }

  // repeat above 3 steps N times and store the result in an OrtValue array
  // create sequence ort value
  OrtValue* seq_ort = nullptr;
  OrtStatus* st = OrtCreateValue(&in[0], N, ONNX_TYPE_SEQUENCE, info, &seq_ort);
  rels.torel.push_back(st);
  ASSERT_EQ(OrtGetErrorCode(st), ORT_OK);

  // Get count
  int num_values;
  OrtStatus* st2 = OrtGetNumValues(seq_ort, &num_values);
  rels.torel.push_back(st2);
  ASSERT_EQ(OrtGetErrorCode(st2), ORT_OK);
  ASSERT_EQ(num_values, N);

  // Fetch
}

// TEST_F(CApiTest, GetVectorOfMaps) {
// }
