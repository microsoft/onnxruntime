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
  int requires_reset = -1;
  float scores[3] = {1.0f, 4.0f, 2.0f};
  const int64_t logits_shape[2] = {1, 3};
  const int64_t cache_shape[4] = {1, 1, 0, 1};
  const int64_t value_shape[4] = {1, 1, 1, 1};
  OrtStatefulSamplerConfig sampler_config = {0};
  OrtStatefulKVCacheSpec spec = {"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, cache_shape, 4, 2, 0};
  OrtStatefulKVCacheConfig cache_config = {0};
  const OrtStatefulProviderSelection cpu = {
      sizeof(cpu), ORT_STATEFUL_PROVIDER_CPU, NULL, OrtMemoryInfoDeviceType_CPU, 0};
  OrtStatefulDispatchInfo dispatch = {0};

  sampler_config.struct_size = sizeof(sampler_config);
  sampler_config.algorithm = ORT_STATEFUL_GREEDY;
  cache_config.struct_size = sizeof(cache_config);
  cache_config.caches = &spec;
  cache_config.num_caches = 1;
  status = api->GetEpApi()->RegisterStatefulProvider(NULL, NULL);
  if (status == NULL) {
    fprintf(stderr, "Registration outside CreateEp must fail\n");
    code = 7;
    goto cleanup;
  }
  api->ReleaseStatus(status);
  status = NULL;
  CHECK(api->GetAllocatorWithDefaultOptions(&allocator));
  CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));
  CHECK(api->CreateTensorWithDataAsOrtValue(memory, scores, sizeof(scores), logits_shape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &logits));
  CHECK(stateful->CreateSamplerOptions(&sampler_config, &sampler));
  CHECK(stateful->SamplerOptionsSetProvider(sampler, NULL, &cpu));
  CHECK(stateful->SamplerSample(sampler, logits, &token));
  CHECK(stateful->SamplerOptionsGetDispatchInfo(sampler, sizeof(dispatch), &dispatch));
  if (dispatch.origin != ORT_STATEFUL_DISPATCH_CPU || dispatch.cpu_calls != 1) {
    code = 5;
    goto cleanup;
  }
  CHECK(api->GetTensorData(logits, &data));
  if (token != 1 || ((const float*)data)[1] != 4.0f) {
    code = 3;
    goto cleanup;
  }
  CHECK(stateful->CreateKVCacheOptions(&cache_config, &cache_options));
  CHECK(stateful->KVCacheOptionsSetProvider(cache_options, NULL, &cpu));
  CHECK(stateful->CreateKVCacheManager(cache_options, allocator, &cache));
  CHECK(api->CreateTensorAsOrtValue(allocator, value_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &cache_tensor));
  CHECK(api->GetTensorMutableData(cache_tensor, &mutable_data));
  *(float*)mutable_data = 3.0f;
  CHECK(stateful->KVCacheManagerUpdate(cache, "key", cache_tensor, ORT_KV_CACHE_SHARE));
  CHECK(stateful->KVCacheManagerGetDispatchInfo(cache, ORT_STATEFUL_CACHE_SHARE, sizeof(dispatch), &dispatch));
  CHECK(stateful->KVCacheManagerRequiresReset(cache, &requires_reset));
  if (dispatch.origin != ORT_STATEFUL_DISPATCH_CPU || dispatch.cpu_calls != 1 || requires_reset != 0) {
    code = 6;
    goto cleanup;
  }
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

int StatefulApiCDeviceConsumerTest(OrtSession* session, OrtAllocator* allocator,
                                   const OrtValue* delta, const OrtValue* full,
                                   OrtKVCacheView** retained_out) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const OrtStatefulApi* stateful = api->GetStatefulApi();
  OrtStatus* status = NULL;
  OrtKVCacheOptions* options = NULL;
  OrtKVCacheManager* cache = NULL;
  OrtKVCacheView* view = NULL;
  const OrtValue* value = NULL;
  int64_t length = -1, capacity = -1;
  int code = 0;
  const int64_t shape[] = {2, 0, 2};
  const OrtStatefulKVCacheSpec spec = {"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 3, 1, 0};
  const OrtStatefulKVCacheConfig config = {sizeof(config), &spec, 1};
  const OrtStatefulProviderSelection selection = {
      sizeof(selection), ORT_STATEFUL_PROVIDER_REQUIRE, "CUDAExecutionProvider", OrtMemoryInfoDeviceType_GPU, 0};
  CHECK(stateful->CreateKVCacheOptions(&config, &options));
  CHECK(stateful->KVCacheOptionsSetProvider(options, session, &selection));
  CHECK(stateful->CreateKVCacheManager(options, allocator, &cache));
  CHECK(stateful->KVCacheManagerUpdate(cache, "key", delta, ORT_KV_CACHE_COPY));
  CHECK(stateful->KVCacheManagerUpdate(cache, "key", delta, ORT_KV_CACHE_APPEND));
  CHECK(stateful->KVCacheManagerReserve(cache, "key", 5));
  CHECK(stateful->KVCacheManagerUpdate(cache, "key", full, ORT_KV_CACHE_SHARE));
  CHECK(stateful->KVCacheManagerGetView(cache, "key", &view));
  CHECK(stateful->KVCacheManagerReset(cache));
  CHECK(stateful->KVCacheViewGetInfo(view, &value, &length, &capacity));
  if (length != 2 || capacity != 2) {
    code = 8;
    goto cleanup;
  }
  *retained_out = view;
  view = NULL;
cleanup:
  stateful->ReleaseKVCacheView(view);
  stateful->ReleaseKVCacheManager(cache);
  stateful->ReleaseKVCacheOptions(options);
  return code;
}

int StatefulApiCGpuLoopIntegrationTest(OrtSession* session, const OrtValue* tokens,
                                       OrtStatefulStepResult** retained_out) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const OrtStatefulApi* stateful = api->GetStatefulApi();
  OrtStatus* status = NULL;
  OrtSessionLoopOptions* options = NULL;
  OrtSessionLoop* loop = NULL;
  OrtStatefulExecution* execution = NULL;
  OrtStatefulStepResult* result = NULL;
  int code = 0;
  const int64_t shape[] = {1, 1, 0, 1};
  const OrtStatefulKVCacheBinding cache = {
      {"key", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 4, 2, 0},
      "past_key",
      "present_key",
      ORT_KV_CACHE_SHARE};
  const char* finite[] = {"token"};
  const char* output[] = {"present_key"};
  const OrtStatefulExitCondition exit = {ORT_STATEFUL_END_OF_INPUTS, 0, NULL, NULL, 0};
  const OrtStatefulProviderSelection gpu = {
      sizeof(gpu), ORT_STATEFUL_PROVIDER_REQUIRE, "CUDAExecutionProvider", OrtMemoryInfoDeviceType_GPU, 0};
  const OrtStatefulNamedValue token = {"token", tokens};
  const OrtStatefulInputBatch batch = {&token, 1};
  OrtStatefulLoopConfig config = {0};
  OrtStatefulInputs inputs = {0};
  config.struct_size = sizeof(config);
  config.finite_input_names = finite;
  config.num_finite_inputs = 1;
  config.output_names = output;
  config.num_outputs = 1;
  config.exit_conditions = &exit;
  config.num_exit_conditions = 1;
  config.kv_caches = &cache;
  config.num_kv_caches = 1;
  inputs.struct_size = sizeof(inputs);
  inputs.batches = &batch;
  inputs.num_batches = 1;
  CHECK(stateful->CreateSessionLoopOptions(&config, &options));
  CHECK(stateful->SessionLoopOptionsSetProvider(options, ORT_STATEFUL_FEATURE_CACHE, &gpu));
  CHECK(stateful->CreateSessionLoop(session, options, &loop));
  CHECK(stateful->CreateLoopExecution(loop, &inputs, &execution));
  CHECK(stateful->StatefulExecutionRun(execution, &result));
  CHECK(stateful->StatefulExecutionReset(execution));
  *retained_out = result;
  result = NULL;
cleanup:
  stateful->ReleaseStatefulStepResult(result);
  stateful->ReleaseStatefulExecution(execution);
  stateful->ReleaseSessionLoop(loop);
  stateful->ReleaseSessionLoopOptions(options);
  return code;
}

#undef CHECK
