// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

struct ApiPointers {
  const OrtApi& ort;
  const OrtEpApi& ep;
  const OrtModelEditorApi& model_editor;
};

/// <summary>
/// Get the global instance of ApiPtrs.
/// </summary>
const ApiPointers& Api();
