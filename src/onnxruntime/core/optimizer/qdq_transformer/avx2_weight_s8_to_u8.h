// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__) || !defined(DISABLE_CONTRIB_OPS)

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @brief Convert constant int8_t weight tensor to uint8_t
 *
 * x86/64 platforms provide better performance computing u8s8 matrix
 * multiplications. Unfortunately AVX2/AVX512 CPUs suffers from value
 * overflow problems
 * https://www.intel.com/content/www/us/en/develop/documentation/onednn-developer-guide-and-reference/top/advanced-topics/nuances-of-int8-computations.html
 *
 * This class identify the s8 constant weight tensors and convert them
 * to u8. We have @Class QDQS8ToU8Transformer that convert
 * activations tensors from s8 to u8. With this, quantized matrix
 * multiplication would be u8u8, avoiding overflow with the cost of
 * performance degradation.
 *
 * If the weight tensor is not constant, then @Class QDQS8ToU8Transformer
 * should have already conver it to u8.
 */
class Avx2WeightS8ToU8Transformer : public GraphTransformer {
 public:
  explicit Avx2WeightS8ToU8Transformer(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("Avx2WeightS8ToU8Transformer", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime

#endif  // x86 or x64
