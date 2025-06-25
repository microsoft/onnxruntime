// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <memory>
#include <unordered_map>

#include "core/optimizer/selectors_actions/selector_action_transformer.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

inline constexpr bool QDQIsInt8Allowed() {
#if !defined(MLAS_TARGET_ARM_ANY)
  return false;
#else
  return true;
#endif
}

/**
Transformer that fuses QDQ and fp32 ops into quantized ops.
*/
class QDQSelectorActionTransformer : public SelectorActionTransformer {
 public:
  QDQSelectorActionTransformer(bool is_int8_allowed,
                               const SatApplyContextVariant& apply_context = {},
                               int64_t qdq_matmulnbits_accuracy_level = 4,
                               concurrency::ThreadPool* intra_op_thread_pool = nullptr);
};

}  // namespace onnxruntime
