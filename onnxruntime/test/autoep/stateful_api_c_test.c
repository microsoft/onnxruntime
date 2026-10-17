// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdio.h>

#include "core/session/onnxruntime_c_api.h"

#define CHECK(call)                                          \
  do {                                                       \
    status = (call);                                         \
    if (status != NULL) {                                    \
      fprintf(stderr, "%s\n", api->GetErrorMessage(status)); \
      api->ReleaseStatus(status);                            \
      code = 1;                                              \
      goto cleanup;                                          \
    }                                                        \
  } while (0)

int StatefulApiCConsumerTest(void) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (api == NULL || api->GetStatefulApi() == NULL) {
    fprintf(stderr, "Stateful API unavailable\n");
    return 2;
  }
  const OrtStatefulApi* stateful = api->GetStatefulApi();
  OrtStatus* status = NULL;
  OrtAllocator* allocator = NULL;
  OrtMemoryInfo* memory = NULL;
  OrtValue* logits = NULL;
  OrtValue* cache_tensor = NULL;
  OrtSamplerOptions* sampler = NULL;
  OrtKVCacheOptions* cache_options = NULL;
  OrtKVCacheManager* cache = NULL;
  OrtKVCacheView* view = NULL;
  const OrtValue* borrowed = NULL;
  const void* data = NULL;
  void* mutable_data = NULL;
  int64_t length = -1, capacity = -1, token = -1;
  int code = 0;
  float scores[3] = {1.0f, 4.0f, 2.0f};
  const int64_t logits_shape[2] = {1, 3};
  const int64_t cache_shape[4] = {1, 1, 0, 1};
  const int64_t value_shape[4] = {1, 1, 1, 1};
  OrtStatefulSamplerConfig sampler_config = {0};
  OrtStatefulKVCacheSpec spec = {"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, cache_shape, 4, 2, 0};
  OrtStatefulKVCacheConfig cache_config = {0};

  sampler_config.struct_size = sizeof(sampler_config);
  sampler_config.algorithm = ORT_STATEFUL_GREEDY;
  cache_config.struct_size = sizeof(cache_config);
  cache_config.caches = &spec;
  cache_config.num_caches = 1;
  CHECK(api->GetAllocatorWithDefaultOptions(&allocator));
  CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));
  CHECK(api->CreateTensorWithDataAsOrtValue(memory, scores, sizeof(scores), logits_shape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &logits));
  CHECK(stateful->CreateSamplerOptions(&sampler_config, &sampler));
  CHECK(stateful->SamplerSample(sampler, logits, &token));
  CHECK(api->GetTensorData(logits, &data));
  if (token != 1 || ((const float*)data)[1] != 4.0f) {
    code = 3;
    goto cleanup;
  }
  CHECK(stateful->CreateKVCacheOptions(&cache_config, &cache_options));
  CHECK(stateful->CreateKVCacheManager(cache_options, allocator, &cache));
  CHECK(api->CreateTensorAsOrtValue(allocator, value_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &cache_tensor));
  CHECK(api->GetTensorMutableData(cache_tensor, &mutable_data));
  *(float*)mutable_data = 3.0f;
  CHECK(stateful->KVCacheManagerUpdate(cache, "key", cache_tensor, ORT_KV_CACHE_SHARE));
  CHECK(stateful->KVCacheManagerGetView(cache, "key", &view));
  stateful->ReleaseKVCacheManager(cache);
  cache = NULL;
  api->ReleaseValue(cache_tensor);
  cache_tensor = NULL;
  CHECK(stateful->KVCacheViewGetInfo(view, &borrowed, &length, &capacity));
  CHECK(api->GetTensorData(borrowed, &data));
  if (length != 1 || capacity != 1 || *(const float*)data != 3.0f) {
    code = 4;
  }

cleanup:
  stateful->ReleaseKVCacheView(view);
  stateful->ReleaseKVCacheManager(cache);
  stateful->ReleaseKVCacheOptions(cache_options);
  stateful->ReleaseSamplerOptions(sampler);
  api->ReleaseValue(cache_tensor);
  api->ReleaseValue(logits);
  api->ReleaseMemoryInfo(memory);
  return code;
}

#undef CHECK
