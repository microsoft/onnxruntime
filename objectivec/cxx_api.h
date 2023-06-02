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
#define HEADER_DIR "onnxruntime/"
#else
#define HEADER_DIR ""
#endif

#ifndef ENABLE_TRAINING_APIS
#include HEADER_DIR "onnxruntime_c_api.h"
#include HEADER_DIR "onnxruntime_cxx_api.h"
#else
#include HEADER_DIR "onnxruntime_training_c_api.h"
#include HEADER_DIR "onnxruntime_training_cxx_api.h"
#endif

#if __has_include(HEADER_DIR "coreml_provider_factory.h")
#define ORT_OBJC_API_COREML_EP_AVAILABLE 1
#include HEADER_DIR "coreml_provider_factory.h"
#else
#define ORT_OBJC_API_COREML_EP_AVAILABLE 0
#endif


#if defined(__clang__)
#pragma clang diagnostic pop
#endif  // defined(__clang__)
