// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OrtCallback {
  void (ORT_API_CALL *f)(void *param) NO_EXCEPTION;
  void *param;
} OrtDeleter;

#ifdef __cplusplus
}
#endif