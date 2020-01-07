// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../../winml/adapter/winml_adapter_c_apis.h"

struct OrtModel {
 public:

 private:
  OrtModel() = default;
  OrtModel(const OrtMapTypeInfo& other) = delete;
  OrtModel& operator=(const OrtMapTypeInfo& other) = delete;
};
