// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <string_view>

namespace onnxruntime {

// The current model versions for saving the ort format models
// This version is NOT onnxruntime version
// Only update this version when there is a file format change which will break the compatibilites
// Once this model version is updated, the kSupportedOrtModelVersions in IsOrtModelVersionSupported
// below will also need to be updated.
// See onnxruntime/core/flatbuffers/schema/README.md for more details on versioning.
// Version 1 - history begins
// Version 2 - add serialization/deserialization of sparse_initializer
// Version 3 - add `graph_doc_string` to Model
// Version 4 - update kernel def hashing to not depend on ordering of type constraint types (NOT BACKWARDS COMPATIBLE)
constexpr const char* kOrtModelVersion = "4";

// Check if the given ort model version is supported in this build
inline bool IsOrtModelVersionSupported(std::string_view ort_model_version) {
  // The ort model versions we will support in this build
  // This may contain more versions than the kOrtModelVersion, based on the compatibilities
  constexpr std::array kSupportedOrtModelVersions{
      kOrtModelVersion,
  };

  const auto it = std::find(kSupportedOrtModelVersions.begin(), kSupportedOrtModelVersions.end(), ort_model_version);
  return it != kSupportedOrtModelVersions.cend();
}

}  // namespace onnxruntime
