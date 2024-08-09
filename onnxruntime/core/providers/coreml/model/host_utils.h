// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file hosts the c++ bridge functions for some host utility functions which
// are available using Objective c only

#pragma once

#include <string>

#if defined(__APPLE__)
// See https://apple.github.io/coremltools/mlmodel/Format/Model.html for the info on each CoreML specification version.
// See https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html for the list of ops
// in each CoreML specification version.

// Specification Versions : OS Availability(Core ML Version)
//
// 4 : iOS 13, macOS 10.15, tvOS 13, watchOS 6 (Core ML 3)
//     - initial version of CoreML EP
// 5 : iOS 14, macOS 11, tvOS 14, watchOS 7 (Core ML 4)
//     - additional layers in NeuralNetwork but currently none are implemented by the CoreML EP
// 6 : iOS 15, macOS 12, tvOS 15, watchOS 8 (Core ML 5)
//     - adds MLProgram (MILSpec.Program)
//     - iOS 15 ops
// 7 : iOS 16, macOS 13, tvOS 16, watchOS 9 (Core ML 6)
//     - iOS 16 ops
// 8 : iOS 17, macOS 14, tvOS 17, watchOS 10 (Core ML 7)
//     - iOS 17 ops
//
// **NOTE** We use the Core ML version not the spec version.
//
// e.g. iOS 13 has Core ML 3 (which is Core ML Specification version 4), and the related macros are
// API_AVAILABLE_COREML3, HAS_COREML3_OR_LATER and onnxruntime::coreml::util::CoreMLVersion() will return 3.

// https://developer.apple.com/documentation/swift/marking-api-availability-in-objective-c
// API_AVAILABLE is used to decorate Objective-C APIs
#define API_AVAILABLE_COREML3 API_AVAILABLE(macos(10.15), ios(13))
#define API_AVAILABLE_COREML4 API_AVAILABLE(macos(11), ios(14))
#define API_AVAILABLE_COREML5 API_AVAILABLE(macos(12), ios(15))
#define API_AVAILABLE_COREML6 API_AVAILABLE(macos(13), ios(16))
#define API_AVAILABLE_COREML7 API_AVAILABLE(macos(14), ios(17))

// @available is used in implementation code
// Base required OS to run CoreML Specification Version 4 (Core ML 3)
#define HAS_COREML3_OR_LATER @available(macOS 10.15, iOS 13, *)
#define HAS_COREML4_OR_LATER @available(macOS 11, iOS 14, *)
#define HAS_COREML5_OR_LATER @available(macOS 12, iOS 15, *)
#define HAS_COREML6_OR_LATER @available(macOS 13, iOS 16, *)
#define HAS_COREML7_OR_LATER @available(macOS 14, iOS 17, *)

#endif

#define MINIMUM_COREML_VERSION 3            // first version we support
#define MINIMUM_COREML_MLPROGRAM_VERSION 5  // first version where ML Program was available

namespace onnxruntime {
namespace coreml {
namespace util {

// Return if we are running on the required OS to enable CoreML EP
// This corresponds to [CoreML Specification Version 4 (Core ML 3)]
bool HasRequiredBaseOS();

// Return the CoreML version if 3 or higher. Otherwise returns -1.
int CoreMLVersion();

// Get a temporary macOS/iOS temp file path
std::string GetTemporaryFilePath();

#if !defined(NDEBUG) && defined(__APPLE__)
// Override location the model is written to so that a) it's easily found and b) it is not automatically deleted
// when the EP exits. Use to debug the model that is generated.
// See onnxruntime/core/providers/coreml/dump_mlprogram_model.py for a script to dump the ML Program.
constexpr const char* kOverrideModelOutputDirectoryEnvVar = "ORT_COREML_EP_MODEL_DIR";
#endif
}  // namespace util
}  // namespace coreml
}  // namespace onnxruntime
