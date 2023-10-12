// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Cuda {

#ifdef USE_CUDA

void RegisterOps(Ort::CustomOpDomain& domain);

#else

void RegisterOps(Ort::CustomOpDomain&) {}

#endif

}  // namespace Cuda