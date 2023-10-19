// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/transpose_optimization/onnx_transpose_optimization.h"

//
// Data structures internal to nhwc transformer implementation.
// Maybe we should use Pimpl Idiom to hide all these into
// an implementation class. But it would add an extra pointer
// chasing during runtime.
//
namespace nhwc_map_internal {

/**
 * @brief For identifying layout sensive operators
 * as candidates for transforming to NHWC ops.
 */
struct OpIdInfo {
  const std::string optype_;
  const std::string domain_;
  const onnxruntime::DataType data_type_;

  OpIdInfo(const std::basic_string_view<char>& op,
           const std::basic_string_view<char>& domain,
           onnxruntime::DataType data_type)
      : optype_(op), domain_(domain), data_type_(data_type) {
  }

  bool operator==(const OpIdInfo& other) const {
    return optype_ == other.optype_ && domain_ == other.domain_ && data_type_ == other.data_type_;
  }
};

/**
 * @brief Hash function for \ref OpIdInfo
 */
class OpIdHash {
 public:
  size_t operator()(const OpIdInfo& op) const {
    size_t h1 = std::hash<std::string>{}(op.optype_);
    size_t h2 = std::hash<std::string>{}(op.domain_);
    size_t h3 = size_t(op.data_type_);
    return h2 ^ (h1 << 4) ^ (h3 << 16);
  }
};

/**
 * @brief Information needed for operator layout transformation
 */
struct OpTransformInfo {
  const std::string optype_;
  const std::string domain_;
  const int version_;
  const bool has_channels_last_attrib_;
};

using OpTransformMap = std::unordered_map<OpIdInfo, OpTransformInfo, OpIdHash>;

}  // namespace nhwc_map_internal

namespace onnxruntime {

/**
@Class NhwcTransformer

Transformer that optimizes the graph by using NHWC nodes instead of NCHW nodes
and inserts nodes to transpose tensors as needed.
*/
class NhwcTransformer : public GraphTransformer {
 private:
 public:
  explicit NhwcTransformer(AllocatorPtr cpu_allocator, std::shared_ptr<KernelRegistry> cpu_kernel_registry) noexcept;

  /**
   * @brief Usually called right after constructor, it shows whether
   *        this transformer should be used under current hardware configuration.
   *
   * @return whether this transformer would be useful under current hardware config
   */
  bool IsActive() {
    return !conv_table_.empty();
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  AllocatorPtr cpu_allocator_;

  /**
   * A mapping table to identify operators that need to be transformed, and map
   * them to the new operators that accept NHWC layout
   */
  nhwc_map_internal::OpTransformMap conv_table_;
};

}  // namespace onnxruntime
