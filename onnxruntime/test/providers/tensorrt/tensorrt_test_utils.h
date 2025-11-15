// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "onnxruntime_cxx_api.h"
#include "core/common/path_string.h"

namespace onnxruntime {
namespace test {
OrtStatus* CreateModelWithTopKWhichContainsGraphOutput(const PathString& model_name);
OrtStatus* CreateModelWithNodeOutputNotUsed(const PathString& model_name);
}  // namespace test
}  // namespace onnxruntime
