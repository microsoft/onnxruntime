// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"

namespace GraphTransformerHelpers {
std::vector<std::pair<std::unique_ptr<onnxruntime::GraphTransformer>, onnxruntime::TransformerLevel>> GetGraphTransformers();
}
