// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "winml_adapter_model.h"
#include "core/graph/onnx_protobuf.h"
#include "ort_apis.h"
#include "../../../winml/adapter/winml_adapter_apis.h"
#include "error_code_helper.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

ORT_API(void, winmla::ReleaseModel, OrtModel* ptr) {
  delete ptr;
}