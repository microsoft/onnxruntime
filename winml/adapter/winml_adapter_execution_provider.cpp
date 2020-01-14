// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "pch.h"
#include "winml_adapter_execution_provider.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

OrtExecutionProvider::OrtExecutionProvider(const std::string provider_id) : provider_id_(std::move(provider_id)) {

}

OrtStatus* OrtExecutionProvider::CreateProvider(const std::string provider_id, OrtExecutionProvider** out) {
  *out = new OrtExecutionProvider(provider_id);
  return nullptr;
}

ORT_API(void, winmla::ReleaseExecutionProvider, OrtExecutionProvider* ptr) {
  delete ptr;
}