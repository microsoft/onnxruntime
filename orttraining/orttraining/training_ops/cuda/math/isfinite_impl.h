// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {
template <typename TSrc>
void IsFinite(cudaStream_t stream, const TSrc* input, bool* output, size_t N);
}
}  // namespace onnxruntime
