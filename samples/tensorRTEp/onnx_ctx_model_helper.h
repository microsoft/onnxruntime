// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
static const std::string EPCONTEXT_OP = "EPContext";

bool GraphHasCtxNode(const OrtGraphViewer* graph_viewer);
}
