// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include "core/session/onnxruntime_c_api.h"

namespace vaip {
void register_xir_ops(const std::vector<OrtCustomOpDomain*>& domains);
}
