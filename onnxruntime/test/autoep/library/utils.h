// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_cxx_api.h"

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

#define RETURN_IF_ERROR(expr) \
  do {                        \
    auto _status = (expr);    \
    if (_status)              \
      return _status;         \
  } while (0)
