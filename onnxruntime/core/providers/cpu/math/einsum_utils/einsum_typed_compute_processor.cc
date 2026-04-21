// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "einsum_typed_compute_processor.h"

namespace onnxruntime {

// Implementations moved to einsum_typed_compute_processor.h to allow header-only
// usage and bypass the shared provider boundary for the CUDA EP.

// Explicit class instantiation
template class EinsumTypedComputeProcessor<float>;
template class EinsumTypedComputeProcessor<int32_t>;
template class EinsumTypedComputeProcessor<double>;
template class EinsumTypedComputeProcessor<int64_t>;
template class EinsumTypedComputeProcessor<MLFloat16>;

}  // namespace onnxruntime
