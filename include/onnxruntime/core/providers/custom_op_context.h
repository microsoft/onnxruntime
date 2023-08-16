// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/session/onnxruntime_cxx_api.h>

// CustomOpContext defines an interface allowing a custom op to access ep-specific resources.
struct CustomOpContext {
  CustomOpContext() = default;
  virtual ~CustomOpContext(){};
  virtual void Init(const OrtKernelContext&){};
};