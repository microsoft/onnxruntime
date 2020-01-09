// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/session/winml_adapter_c_api.h"

struct OrtModel {
 public:
  static OrtStatus* CreateOrtModelFromPath(const char* path, size_t len, OrtModel** model);
  static OrtStatus* CreateOrtModelFromData(void* data, size_t len, OrtModel** model); 

 private:
  OrtModel() = default;
  OrtModel(const OrtModel& other) = delete;
  OrtModel& operator=(const OrtModel& other) = delete;
};
