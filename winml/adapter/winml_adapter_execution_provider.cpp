// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "pch.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

ORT_API_STATUS_IMPL(winmla::ExecutionProviderSync, _In_ const OrtExecutionProvider* provider) {
  API_IMPL_BEGIN
  const auto execution_provider = reinterpret_cast<const onnxruntime::IExecutionProvider*>(provider);
  execution_provider->Sync();
  return nullptr;
  API_IMPL_END
}