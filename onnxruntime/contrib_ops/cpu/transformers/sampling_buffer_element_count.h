// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/safeint.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// Computes the element count for a `SamplingState` buffer using
// `SafeMul<size_t>` so that overflow (from a plain `int * int` multiply) or
// negative-to-unsigned conversion (e.g. an unvalidated `-1`) of the
// model-controlled operands is detected up front (throws
// `OnnxRuntimeException`) rather than silently producing an under-sized or
// absurdly large allocation. Extracted into a self-contained header so that
// both `SamplingState::Init` and its regression tests call the exact same
// production code path; reverting this computation to `int * int` or
// `static_cast<size_t>(...)` will fail the tests.
inline size_t SamplingBufferElementCount(int batch_size, int per_batch_count) {
  return SafeMul<size_t>(batch_size, per_batch_count);
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
