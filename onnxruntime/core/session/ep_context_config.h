// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

struct OrtEpContextConfig {
  OrtWriteNamedBufferFunc write_func = nullptr;
  void* write_state = nullptr;
  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
};