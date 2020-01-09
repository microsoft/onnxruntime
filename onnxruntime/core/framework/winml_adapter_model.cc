// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "winml_adapter_model.h"

#include "core/session/winml_adapter_c_api.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/session/winml_adapter_apis.h"
#include "error_code_helper.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

OrtStatus* OrtModel::CreateOrtModelFromPath(const char* path, size_t len, OrtModel** model) {
  ORT_UNUSED_PARAMETER(path);
  ORT_UNUSED_PARAMETER(len);
  ORT_UNUSED_PARAMETER(model);
  return nullptr;
}

OrtStatus* OrtModel::CreateOrtModelFromData(void* data, size_t len, OrtModel** model) {
  ORT_UNUSED_PARAMETER(data);
  ORT_UNUSED_PARAMETER(len);
  ORT_UNUSED_PARAMETER(model);
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::CreateModelFromPath, const char* model_path, _In_ size_t size, _Outptr_ OrtModel** out) {
  if (auto status = OrtModel::CreateOrtModelFromPath(model_path, size, out))
  {
    return status;
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::CreateModelFromData, void* data, _In_ size_t size, _Outptr_ OrtModel** out) {
  if (auto status = OrtModel::CreateOrtModelFromData(data, size, out)) {
    return status;
  }
  return nullptr;
}

ORT_API(void, winmla::ReleaseModel, OrtModel* ptr) {
  delete ptr;
}