// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@class ConstantSharing

Transformer that traverses the graph top-down and performs constant sharing, i.e.,
For constant initializers that share same values and shapes, share one single initializer, and remove others.
Currently, only handle those scalar valued initializers.
*/
class ConstantSharing : public GraphTransformer {
 public:
  /**
   * @param compatible_execution_providers comptatible execution provider list for considered nodes.
   * @param excluded_initializers explicitly excluded initializer names that should not changed.
   */
  ConstantSharing(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                  const InlinedHashSet<std::string>& excluded_initializers = {}) noexcept
      : GraphTransformer("ConstantSharing", compatible_execution_providers),
        excluded_initializers_(excluded_initializers) {
  }

 private:
  template <typename T>
  bool ShareInitializer(Graph& graph, Node* node, int input_index,
                        std::map<std::tuple<int32_t, int64_t, int>, NodeArg*>&
                            type_value_plus_rank_to_shared_arg_map,
                        int32_t data_type) const;

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  const InlinedHashSet<std::string> excluded_initializers_;
};

}  // namespace onnxruntime
