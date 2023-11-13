// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Cuda {

#if defined(USE_CUDA) && !defined(ENABLE_TRAINING)

void RegisterOps(Ort::CustomOpDomain& domain);

#else

void RegisterOps(Ort::CustomOpDomain&) {}

#endif

}  // namespace Cuda