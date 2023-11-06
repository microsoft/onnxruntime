// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

#include <framework/custom_ep.h>

using namespace onnxruntime::test;

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_API CustomEp* GetExternalProvider(const void*) {
  CustomEpInfo info;
  return std::make_unique<CustomEp>(info).release();
}

#ifdef __cplusplus
}
#endif