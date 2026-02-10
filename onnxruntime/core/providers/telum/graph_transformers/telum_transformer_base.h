// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "../telum_common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Base class for Telum-specific graph transformers
 *
 * Provides common functionality for pattern matching and fusion
 * in transformer models. All Telum graph transformers should inherit
 * from this class.
 */
class TelumTransformerBase : public GraphTransformer {
 public:
  TelumTransformerBase(const std::string& name,
                       const InlinedHashSet<std::string_view>& compatible_eps = {})
      : GraphTransformer(name, compatible_eps) {}

 protected:
  /**
   * @brief Check if node has static shapes
   */
  bool HasStaticShapes(const Node& node) const {
    for (const auto* input_def : node.InputDefs()) {
      if (input_def == nullptr) {
        continue;  // optional input not provided
      }
      if (input_def->Shape() == nullptr) {
        return false;
      }
      const auto& shape = *input_def->Shape();
      for (const auto& dim : shape.dim()) {
        if (!dim.has_dim_value()) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * @brief Check if node output is consumed by specific op type
   */
  bool IsConsumedBy(const Node& node, const std::string& op_type) const {
    for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
      if (it->GetNode().OpType() == op_type) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Get single consumer of node output
   * @return Pointer to consumer node, or nullptr if multiple or no consumers
   */
  const Node* GetSingleConsumer(const Node& node) const {
    const Node* consumer = nullptr;
    for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
      if (consumer != nullptr) {
        return nullptr;  // Multiple consumers
      }
      consumer = &it->GetNode();
    }
    return consumer;
  }

  /**
   * @brief Check if two nodes are in a producer-consumer relationship
   */
  bool IsProducerConsumer(const Node& producer, const Node& consumer) const {
    for (auto it = producer.OutputEdgesBegin(); it != producer.OutputEdgesEnd(); ++it) {
      if (&it->GetNode() == &consumer) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Get tensor shape from node output
   */
  std::vector<int64_t> GetShape(const NodeArg* node_arg) const {
    std::vector<int64_t> shape;
    if (node_arg && node_arg->Shape()) {
      const auto& tensor_shape = *node_arg->Shape();
      for (const auto& dim : tensor_shape.dim()) {
        if (dim.has_dim_value()) {
          shape.push_back(dim.dim_value());
        } else {
          return {};  // Dynamic shape
        }
      }
    }
    return shape;
  }

  /**
   * @brief Check if shape matches expected dimensions
   */
  bool ShapeMatches(const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& expected) const {
    if (shape.size() != expected.size()) {
      return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
      if (expected[i] != -1 && shape[i] != expected[i]) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
