// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#include "ort_instance_data.h"
#include "ort_singleton_data.h"
#include "onnxruntime_cxx_api.h"

OrtInstanceData::OrtInstanceData() {
}

void OrtInstanceData::Create(Napi::Env env, Napi::Function inferenceSessionWrapperFunction) {
  ORT_NAPI_THROW_ERROR_IF(env.GetInstanceData<void>() != nullptr, env, "OrtInstanceData already created.");
  auto data = new OrtInstanceData{};
  data->wrappedSessionConstructor = Napi::Persistent(inferenceSessionWrapperFunction);
  env.SetInstanceData(data);
}

void OrtInstanceData::InitOrt(Napi::Env env, int log_level, Napi::Function tensorConstructor, bool is_main_thread) {
  auto data = env.GetInstanceData<OrtInstanceData>();
  ORT_NAPI_THROW_ERROR_IF(data == nullptr, env, "OrtInstanceData not created.");

  data->ortTensorConstructor = Napi::Persistent(tensorConstructor);

  // Initialize ORT singleton and register cleanup hook for this env.
  // The first call creates the OrtObjects; subsequent calls increment the ref count.
  OrtSingletonData::InitOrtObjects(env, log_level, is_main_thread);
}

const Napi::FunctionReference& OrtInstanceData::TensorConstructor(Napi::Env env) {
  auto data = env.GetInstanceData<OrtInstanceData>();
  ORT_NAPI_THROW_ERROR_IF(data == nullptr, env, "OrtInstanceData not created.");

  return data->ortTensorConstructor;
}
