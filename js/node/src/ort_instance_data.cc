// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <mutex>

#include "common.h"
#include "ort_instance_data.h"
#include "onnxruntime_cxx_api.h"

std::unique_ptr<Ort::Env> OrtInstanceData::ortEnv;
std::unique_ptr<Ort::RunOptions> OrtInstanceData::ortDefaultRunOptions;
std::mutex OrtInstanceData::ortEnvMutex;
std::atomic<uint64_t> OrtInstanceData::ortEnvRefCount;
std::atomic<bool> OrtInstanceData::ortEnvDestroyed;

OrtInstanceData::OrtInstanceData() {
  ++ortEnvRefCount;
}

OrtInstanceData::~OrtInstanceData() {
  if (--ortEnvRefCount == 0) {
    std::lock_guard<std::mutex> lock(ortEnvMutex);
    if (ortEnv) {
      ortDefaultRunOptions.reset(nullptr);
      ortEnv.reset();
      ortEnvDestroyed = true;
    }
  }
}

void OrtInstanceData::Create(Napi::Env env, Napi::Function inferenceSessionWrapperFunction) {
  ORT_NAPI_THROW_ERROR_IF(env.GetInstanceData<void>() != nullptr, env, "OrtInstanceData already created.");
  auto data = new OrtInstanceData{};
  data->wrappedSessionConstructor = Napi::Persistent(inferenceSessionWrapperFunction);
  env.SetInstanceData(data);
}

void OrtInstanceData::InitOrt(Napi::Env env, int log_level, Napi::Function tensorConstructor) {
  auto data = env.GetInstanceData<OrtInstanceData>();
  ORT_NAPI_THROW_ERROR_IF(data == nullptr, env, "OrtInstanceData not created.");

  data->ortTensorConstructor = Napi::Persistent(tensorConstructor);

  if (!ortEnv) {
    std::lock_guard<std::mutex> lock(ortEnvMutex);
    if (!ortEnv) {
      ORT_NAPI_THROW_ERROR_IF(ortEnvDestroyed, env, "OrtEnv already destroyed.");
      ortEnv.reset(new Ort::Env{OrtLoggingLevel(log_level), "onnxruntime-node"});
      ortDefaultRunOptions.reset(new Ort::RunOptions{});
    }
  }
}

const Napi::FunctionReference& OrtInstanceData::TensorConstructor(Napi::Env env) {
  auto data = env.GetInstanceData<OrtInstanceData>();
  ORT_NAPI_THROW_ERROR_IF(data == nullptr, env, "OrtInstanceData not created.");

  return data->ortTensorConstructor;
}
