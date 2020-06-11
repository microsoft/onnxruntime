// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace onnxruntime {

bool IsOperationDeterministic(const std::string& domain, const std::string& op);

}
