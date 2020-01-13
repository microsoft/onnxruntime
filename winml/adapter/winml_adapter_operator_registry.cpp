// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "pch.h"

#include "winml_adapter_operator_registry.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

ORT_API(void, winmla::ReleaseOperatorRegistry, OrtOperatorRegistry* ptr) {
  delete ptr;
}