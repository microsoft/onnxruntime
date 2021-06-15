// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
    
namespace training {

void RegisterTrainingOpSchemas();

}  // namespace training

// Top-level function for registering all contrib and training op schemas, as well as their domains.
void RegisterOrtOpSchemas();

}  // namespace onnxruntime
