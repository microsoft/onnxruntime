// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>

namespace onnxruntime {

const std::unordered_set<std::string> kNonDeterministicOps =
      {"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Multinomial"};

}
