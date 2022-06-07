// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file hosts the c++ bridge functions for some host utility functions which
// are available using Objective c only

#pragma once

#define API_AVAILABLE_OS_VERSIONS API_AVAILABLE(macos(10.15), ios(13))

// Base requireed OS to run CoreML Specification Version 4 (Core ML 3)
#define HAS_VALID_BASE_OS_VERSION @available(macOS 10.15, iOS 13, *)

namespace onnxruntime {
namespace coreml {
namespace util {

// Return if we are running on the required OS to enable CoreML EP
// This corresponds to [CoreML Specification Version 4 (Core ML 3)]
bool HasRequiredBaseOS();

// Get a temporary macOS/iOS temp file path
std::string GetTemporaryFilePath();

}  // namespace util
}  // namespace coreml
}  // namespace onnxruntime
