// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Rocm {

#ifdef USE_ROCM

void RegisterOps(Ort::CustomOpDomain& domain);

#else

inline void RegisterOps(Ort::CustomOpDomain&) {}

#endif

}  // namespace Rocm
