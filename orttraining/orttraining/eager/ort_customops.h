// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ort_eager_common.h"

namespace torch_ort {
namespace eager {

void GenerateCustomOpsBindings(pybind11::module_ module);

} // namespace eager
} // namespace torch_ort