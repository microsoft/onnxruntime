// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

/**
@class ConstantSharing

Transformer that traverses the graph top-down and performs constant sharing, i.e.,
constant initializers having same data type, value and shape, will be replaced by one single initializer.
Currently, only scalar-valued initializers are handled.
*/
class ConstantSharing : public GraphTransformer {
 public:
  /**
   * @param compatible_execution_providers compatible execution provider list for considered nodes.
   * @param excluded_initializers explicitly excluded initializer names that should not changed.
   */
  ConstantSharing(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                  const InlinedHashSet<std::string>& excluded_initializers = {}) noexcept
      : GraphTransformer("ConstantSharing", compatible_execution_providers),
        excluded_initializers_(excluded_initializers) {
  }

  bool ShouldOnlyApplyOnce() const override {
#if defined(ENABLE_TRAINING)
    return false;
#else
    // Reduce model processing time by applying this optimization only once for inference.
    return true;
#endif
  }

  static constexpr int64_t TENSOR_ELEM_COUNT_THRESHOLD = 8;

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  const InlinedHashSet<std::string> excluded_initializers_;
};

}  // namespace onnxruntime
