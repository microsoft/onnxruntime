// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_options.h"

namespace onnxruntime {

ModelPackageOptions::ModelPackageOptions(const OrtSessionOptions& session_options)
    : session_options_(session_options) {}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)