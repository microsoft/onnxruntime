// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// wrapper for ORT C/C++ API headers

#if defined(__clang__)
#pragma clang diagnostic push
// ignore clang documentation-related warnings
// instead, we will rely on Doxygen warnings for the C/C++ API headers
#pragma clang diagnostic ignored "-Wdocumentation"
#endif  // defined(__clang__)

// paths are different when building the Swift Package Manager package as the headers come from the iOS pod archive
#ifdef SPM_BUILD
#include "onnxruntime/onnxruntime_c_api.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

#if __has_include("onnxruntime/coreml_provider_factory.h")
#define ORT_OBJC_API_COREML_EP_AVAILABLE 1
#include "onnxruntime/coreml_provider_factory.h"
#else
#define ORT_OBJC_API_COREML_EP_AVAILABLE 0
#endif

#else
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#if __has_include("coreml_provider_factory.h")
#define ORT_OBJC_API_COREML_EP_AVAILABLE 1
#include "coreml_provider_factory.h"
#else
#define ORT_OBJC_API_COREML_EP_AVAILABLE 0

#endif

#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif  // defined(__clang__)
