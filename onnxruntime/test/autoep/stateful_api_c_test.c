// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdio.h>
#include <string.h>

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

int StatefulApiCDdimConsumer(OrtSession* session, int sequential, const OrtValue* init,
                             const OrtValue* positive, const OrtValue* negative,
                             OrtStatefulStepResult** retained_out) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const OrtStatefulApi* stateful = api->GetStatefulApi();
  OrtStatus* status = NULL;
  OrtSessionLoopOptions* options = NULL;
  OrtSessionLoop* loop = NULL;
  OrtStatefulExecution* execution = NULL;
  OrtStatefulStepResult* first = NULL;
  OrtStatefulStepResult* result = NULL;
  OrtStatefulLoopConfig config = {0};
  OrtStatefulDdimConfig ddim = {0};
  OrtStatefulInputs inputs = {0};
  OrtStatefulStepInfo info = {0};
  const int64_t shape[4] = {1, 1, 1, 3};
  const char* output_names[1] = {"latents"};
  const OrtStatefulExitCondition exit = {ORT_STATEFUL_MAX_ITERATIONS, 2, NULL, NULL, 0};
  const OrtStatefulNamedValue values[3] = {{"init", init}, {"positive", positive}, {"negative", negative}};
  int code = 0;
  config.struct_size = sizeof(config);
  config.output_names = output_names;
  config.num_outputs = 1;
  config.exit_conditions = &exit;
  config.num_exit_conditions = 1;
  ddim.struct_size = sizeof(ddim);
  ddim.profile = ORT_STATEFUL_DDIM_SD15_V1;
  ddim.latent_shape = shape;
  ddim.num_dimensions = 4;
  ddim.num_inference_steps = 2;
  ddim.guidance_scale = 7.5f;
  ddim.guidance_mode = sequential ? ORT_STATEFUL_DDIM_GUIDANCE_SEQUENTIAL : ORT_STATEFUL_DDIM_GUIDANCE_BATCHED;
  ddim.timestep_format = ORT_STATEFUL_DDIM_TIMESTEP_FLOAT32_SCALAR;
  ddim.latent_input = "sample";
  ddim.timestep_input = "time";
  ddim.conditioning_input = "conditioning";
  ddim.prediction_output = "epsilon";
  ddim.initialization_input = "init";
  ddim.positive_conditioning_input = "positive";
  ddim.negative_conditioning_input = "negative";
  ddim.latent_output = "latents";
  CHECK(stateful->CreateSessionLoopOptions(&config, &options));
  CHECK(stateful->SessionLoopOptionsSetDdim(options, &ddim));
  CHECK(stateful->CreateSessionLoop(session, options, &loop));
  CHECK(stateful->SessionLoopOptionsSetDdim(options, NULL));
  inputs.struct_size = sizeof(inputs);
  inputs.inputs = values;
  inputs.num_inputs = 3;
  CHECK(stateful->CreateLoopExecution(loop, &inputs, &execution));
  CHECK(stateful->StatefulExecutionStep(execution, &first));
  CHECK(stateful->StatefulStepResultGetInfo(first, sizeof(info), &info));
  if (!info.executed || info.completed || info.iterations != 1 || info.invocations != (sequential ? 2 : 1)) {
    fprintf(stderr, "Incorrect C11 DDIM first-step progress\n");
    code = 60;
    goto cleanup;
  }
  CHECK(stateful->StatefulExecutionRun(execution, &result));
  CHECK(stateful->StatefulStepResultGetInfo(result, sizeof(info), &info));
  if (!info.completed || info.iterations != 2 || info.invocations != (sequential ? 4 : 2) ||
      info.generated_tokens != 0 || info.has_generated_token) {
    fprintf(stderr, "Incorrect C11 DDIM completion progress\n");
    code = 61;
    goto cleanup;
  }
  CHECK(stateful->StatefulExecutionReset(execution));
  if (retained_out != NULL) {
    *retained_out = result;
    result = NULL;
  }
cleanup:
  stateful->ReleaseStatefulStepResult(result);
  stateful->ReleaseStatefulStepResult(first);
  stateful->ReleaseStatefulExecution(execution);
  stateful->ReleaseSessionLoop(loop);
  stateful->ReleaseSessionLoopOptions(options);
  return code;
}

static int StatefulApiCPackageChainTestImpl(const OrtEnv* env, const ORTCHAR_T* path,
                                            const OrtSessionOptions* prefill_options,
                                            const OrtSessionOptions* decode_options,
                                            OrtStatefulStepResult** retained_out) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const OrtStatefulApi* stateful = api->GetStatefulApi();
  const OrtModelPackageApi* package = api->GetModelPackageApi();
  OrtStatus* status = NULL;
  OrtModelPackageContext* context = NULL;
  OrtSessionOptions* options = NULL;
  OrtSessionChain* chain = NULL;
  OrtStatefulExecution* execution = NULL;
  OrtStatefulStepResult* first = NULL;
  OrtStatefulStepResult* sampled = NULL;
  OrtStatefulStepResult* result = NULL;
  OrtValue* prompt = NULL;
  OrtAllocator* allocator = NULL;
  OrtTensorTypeAndShapeInfo* shape_info = NULL;
  OrtStatefulGenerationBootstrapConfig bootstrap = {0};
  OrtModelPackageChainConfig config = {0};
  OrtStatefulInputs inputs = {0};
  OrtStatefulStepInfo info = {0};
  const OrtValue* values[1] = {NULL};
  const OrtStatefulInputSequence sequence = {"prompt_batches", values, 1};
  const OrtStatefulNamedValue* outputs = NULL;
  const OrtValue* token_output = NULL;
  OrtStatefulDispatchInfo dispatch = {0};
  size_t count = 0;
  void* writable = NULL;
  const void* data = NULL;
  const int64_t shape[2] = {1, 2};
  OrtModelPackageStepSessionOptions overrides[2] = {{0}, {0}};
  int code = 0;
  bootstrap.struct_size = sizeof(bootstrap);
  status = stateful->SessionChainOptionsSetGenerationBootstrap(NULL, 0, &bootstrap);
  if (status == NULL) {
    fprintf(stderr, "NULL bootstrap options unexpectedly succeeded\n");
    code = 30;
    goto cleanup;
  }
  api->ReleaseStatus(status);
  status = NULL;
  CHECK(api->CreateSessionOptions(&options));
  CHECK(api->SetIntraOpNumThreads(options, 1));
  CHECK(api->SetInterOpNumThreads(options, 1));
  CHECK(package->CreateModelPackageContext(path, &context));
  overrides[0].step_id = "prefill";
  overrides[1].step_id = "decode";
  overrides[0].session_options = prefill_options ? prefill_options : options;
  overrides[1].session_options = decode_options ? decode_options : options;
  config.struct_size = sizeof(config);
  config.chain_name = "generate";
  config.step_session_options = overrides;
  config.num_step_session_options = 2;
  CHECK(package->CreateSessionChain(env, context, &config, &chain));
  package->ReleaseModelPackageContext(context);
  context = NULL;
  api->ReleaseSessionOptions(options);
  options = NULL;
  CHECK(api->GetAllocatorWithDefaultOptions(&allocator));
  CHECK(api->CreateTensorAsOrtValue(allocator, shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &prompt));
  CHECK(api->GetTensorMutableData(prompt, &writable));
  ((int64_t*)writable)[0] = 0;
  ((int64_t*)writable)[1] = 1;
  values[0] = prompt;
  inputs.struct_size = sizeof(inputs);
  inputs.sequences = &sequence;
  inputs.num_sequences = 1;
  CHECK(stateful->CreateChainExecution(chain, &inputs, &execution));
  stateful->ReleaseSessionChain(chain);
  chain = NULL;
  CHECK(stateful->StatefulExecutionStep(execution, &first));
  CHECK(stateful->StatefulStepResultGetInfo(first, sizeof(info), &info));
  if (!info.executed || info.invocations != 1 || info.generated_tokens != 0) {
    fprintf(stderr, "Unexpected prefill progress in C11 consumer\n");
    code = 31;
    goto cleanup;
  }
  CHECK(stateful->StatefulExecutionStep(execution, &sampled));
  CHECK(stateful->StatefulStepResultGetInfo(sampled, sizeof(info), &info));
  if (info.executed || info.iterations != 0 || info.invocations != 1 || !info.has_generated_token ||
      info.generated_token != 2) {
    fprintf(stderr, "Unexpected sample-only progress in C11 consumer\n");
    code = 32;
    goto cleanup;
  }
  CHECK(stateful->StatefulExecutionRun(execution, &result));
  CHECK(stateful->StatefulStepResultGetOutputs(result, ORT_STATEFUL_SELECTED_OUTPUTS, &outputs, &count));
  for (size_t i = 0; i < count; ++i) {
    if (strcmp(outputs[i].name, "tokens") == 0) token_output = outputs[i].value;
  }
  if (token_output == NULL) {
    fprintf(stderr, "Missing selected token output in C11 consumer\n");
    code = 33;
    goto cleanup;
  }
  CHECK(api->GetTensorTypeAndShape(token_output, &shape_info));
  CHECK(api->GetTensorShapeElementCount(shape_info, &count));
  CHECK(api->GetTensorData(token_output, &data));
  if (count != 3 || ((const int64_t*)data)[0] != 2 || ((const int64_t*)data)[1] != 3 ||
      ((const int64_t*)data)[2] != 0) {
    fprintf(stderr, "Incorrect C11 package token history\n");
    code = 34;
    goto cleanup;
  }
  if (retained_out) {
    CHECK(stateful->StatefulExecutionGetDispatchInfo(execution, ORT_STATEFUL_FEATURE_SAMPLER, 1,
                                                     ORT_STATEFUL_SAMPLE_GREEDY, sizeof(dispatch), &dispatch));
    if (dispatch.origin != ORT_STATEFUL_DISPATCH_PROVIDER || dispatch.device_type != OrtMemoryInfoDeviceType_GPU ||
        dispatch.provider_calls != 3 || dispatch.cpu_calls != 0 || dispatch.cpu_delegations != 0) {
      fprintf(stderr, "C11 package did not execute native GPU greedy\n");
      code = 35;
      goto cleanup;
    }
    CHECK(stateful->StatefulExecutionReset(execution));
    *retained_out = result;
    result = NULL;
  }
cleanup:
  api->ReleaseTensorTypeAndShapeInfo(shape_info);
  stateful->ReleaseStatefulStepResult(first);
  stateful->ReleaseStatefulStepResult(sampled);
  stateful->ReleaseStatefulStepResult(result);
  stateful->ReleaseStatefulExecution(execution);
  stateful->ReleaseSessionChain(chain);
  api->ReleaseValue(prompt);
  api->ReleaseSessionOptions(options);
  package->ReleaseModelPackageContext(context);
  return code;
}

int StatefulApiCPackageChainTest(const OrtEnv* env, const ORTCHAR_T* path) {
  return StatefulApiCPackageChainTestImpl(env, path, NULL, NULL, NULL);
}

int StatefulApiCPackageGpuChainTest(const OrtEnv* env, const ORTCHAR_T* path, const OrtSessionOptions* prefill_options,
                                    const OrtSessionOptions* decode_options, OrtStatefulStepResult** retained_out) {
  return StatefulApiCPackageChainTestImpl(env, path, prefill_options, decode_options, retained_out);
}

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

int StatefulApiCGpuLoopIntegrationTest(OrtSession* session, const OrtValue* tokens, int gpu_sampler,
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
  const char* output[] = {"present_key", "logits"};
  const OrtStatefulExitCondition exit = {ORT_STATEFUL_END_OF_INPUTS, 0, NULL, NULL, 0};
  const OrtStatefulProviderSelection gpu = {
      sizeof(gpu), ORT_STATEFUL_PROVIDER_REQUIRE, "CUDAExecutionProvider", OrtMemoryInfoDeviceType_GPU, 0};
  const OrtStatefulNamedValue token = {"token", tokens};
  const OrtStatefulInputBatch batch = {&token, 1};
  const OrtStatefulGenerationConfig generation = {
      sizeof(generation), ORT_STATEFUL_PREFILL, "token", "logits", NULL, NULL, 0, 0};
  OrtStatefulStepInfo info = {0};
  OrtStatefulLoopConfig config = {0};
  OrtStatefulInputs inputs = {0};
  config.struct_size = sizeof(config);
  config.finite_input_names = finite;
  config.num_finite_inputs = 1;
  config.output_names = output;
  config.num_outputs = gpu_sampler ? 2 : 1;
  config.exit_conditions = &exit;
  config.num_exit_conditions = 1;
  config.kv_caches = &cache;
  config.num_kv_caches = 1;
  config.generation = gpu_sampler ? &generation : NULL;
  inputs.struct_size = sizeof(inputs);
  inputs.batches = &batch;
  inputs.num_batches = 1;
  CHECK(stateful->CreateSessionLoopOptions(&config, &options));
  CHECK(stateful->SessionLoopOptionsSetProvider(options, ORT_STATEFUL_FEATURE_CACHE, &gpu));
  if (gpu_sampler) CHECK(stateful->SessionLoopOptionsSetProvider(options, ORT_STATEFUL_FEATURE_SAMPLER, &gpu));
  CHECK(stateful->CreateSessionLoop(session, options, &loop));
  CHECK(stateful->CreateLoopExecution(loop, &inputs, &execution));
  CHECK(stateful->StatefulExecutionRun(execution, &result));
  CHECK(stateful->StatefulStepResultGetInfo(result, sizeof(info), &info));
  if (gpu_sampler && (!info.has_generated_token || info.generated_token != 2)) {
    code = 10;
    goto cleanup;
  }
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

int StatefulApiCGreedyDeviceConsumerTest(OrtSession* session, const OrtValue* logits, int64_t* token) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const OrtStatefulApi* stateful = api->GetStatefulApi();
  OrtStatus* status = NULL;
  OrtSamplerOptions* sampler = NULL;
  int code = 0;
  const OrtStatefulSamplerConfig config = {sizeof(config), ORT_STATEFUL_GREEDY, 0, 0};
  const OrtStatefulProviderSelection choice = {
      sizeof(choice), ORT_STATEFUL_PROVIDER_REQUIRE, "CUDAExecutionProvider", OrtMemoryInfoDeviceType_GPU, 0};
  OrtStatefulDispatchInfo info = {0};
  CHECK(stateful->CreateSamplerOptions(&config, &sampler));
  CHECK(stateful->SamplerOptionsSetProvider(sampler, session, &choice));
  CHECK(stateful->SamplerSample(sampler, logits, token));
  CHECK(stateful->SamplerOptionsGetDispatchInfo(sampler, sizeof(info), &info));
  if (info.provider_calls != 1 || info.cpu_calls != 0 || info.cpu_delegations != 0) code = 9;
cleanup:
  stateful->ReleaseSamplerOptions(sampler);
  return code;
}

#undef CHECK
