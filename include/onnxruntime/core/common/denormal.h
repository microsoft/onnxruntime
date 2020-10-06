// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

bool SetDenormalAsZero(bool on);

#ifdef _OPENMP
void InitializeWithDenormalAsZero(bool on);
#endif

}  // namespace onnxruntime
