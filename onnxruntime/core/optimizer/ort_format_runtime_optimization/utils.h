// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime::optimizer_utils {

/**
 * Generates the ORT format model runtime graph optimization transformers.
 */
std::vector<std::unique_ptr<GraphTransformer>> GenerateOrtFormatRuntimeTransformers();

}  // namespace onnxruntime::optimizer_utils
