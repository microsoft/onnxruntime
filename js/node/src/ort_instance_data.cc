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

namespace {
bool RegionsOverlap(size_t offset, size_t length, size_t other_offset, size_t other_length) {
  // A zero length means "the whole resource", so it conflicts with any other region.
  if (length == 0 || other_length == 0) {
    return true;
  }
  return offset < other_offset + other_length && other_offset < offset + length;
}
}  // namespace

OrtInstanceData::OutputBufferLease OrtInstanceData::AcquireOutputBufferLease(Napi::Object resource,
                                                                            size_t byteOffset, size_t byteLength) {
  auto data = resource.Env().GetInstanceData<OrtInstanceData>();
  ORT_NAPI_THROW_ERROR_IF(data == nullptr, resource.Env(), "OrtInstanceData not created.");

  for (const auto& lease : data->outputBufferLeases) {
    ORT_NAPI_THROW_ERROR_IF(lease->resource.Value().StrictEquals(resource) &&
                                RegionsOverlap(byteOffset, byteLength, lease->byteOffset, lease->byteLength),
                            resource.Env(), "Preallocated output buffer is already in use.");
  }

  auto lease = std::make_shared<OutputBufferRegion>(
      OutputBufferRegion{Napi::Persistent(resource), byteOffset, byteLength});
  data->outputBufferLeases.push_back(lease);
  return lease;
}

void OrtInstanceData::ReleaseOutputBufferLease(Napi::Env env, const OutputBufferLease& lease) {
  auto data = env.GetInstanceData<OrtInstanceData>();
  if (data == nullptr) {
    return;
  }

  for (auto it = data->outputBufferLeases.begin(); it != data->outputBufferLeases.end(); ++it) {
    if (*it == lease) {
      data->outputBufferLeases.erase(it);
      return;
    }
  }
}
