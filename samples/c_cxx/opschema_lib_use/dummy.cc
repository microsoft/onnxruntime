// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

// Dummy implementations of dependencies used by op schema library (for testing purpose).

namespace onnxruntime {

std::vector<std::string> GetStackTrace() { return {}; }

namespace math {

uint16_t floatToHalf(float f) { return 0; }

}  // namespace math

}  // namespace onnxruntime

size_t MlasNchwcGetBlockSize() { return 0; }
