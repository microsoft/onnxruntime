// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include "gsl/gsl"
#include "core/framework/allocator.h"
#include "core/framework/feeds_fetches_manager.h"
#include "contrib_ops/cpu/transformers/beam_search_device_helper.h"

namespace onnxruntime {
class SessionState;
}

namespace onnxruntime {
namespace contrib {
namespace transformers {

class Subgraph {
 public:
  Subgraph(
      const onnxruntime::Node& node_in,
      const std::string& attribute_name,
      const GraphViewer& subgraph_in);
  virtual ~Subgraph() {}

  const onnxruntime::Node& node;  // Node that contains the subgraph
  const std::string& attribute;   // Attribute of th node that contains the subgraph. Not used yet.
  const GraphViewer& subgraph;    // The subgraph

  int num_implicit_inputs;

  int num_subgraph_inputs;   // Same as subgraph_input_names.size(), keep it for convenience.
  int num_subgraph_outputs;  // Same as subgraph_output_names.size()

  std::vector<std::string> subgraph_input_names;
  std::vector<std::string> subgraph_output_names;

  // Parameters deduced from the subgraph
  int num_heads;
  int head_size;
  int vocab_size;
  int num_layers;

  // Setup execution
  Status Setup(const SessionState& session_state,
               const SessionState& subgraph_session_state);

  FeedsFetchesManager* GetFeedsFetchesManager() const { return feeds_fetches_manager_.get(); }

  const IExecutionProvider* GetProvider() const;

  bool IsOutputFloat16() const { return is_output_float16_; }

  virtual Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                          const std::vector<const NodeArg*>& subgraph_outputs) = 0;

 protected:
  Status GetParameters(const ONNX_NAMESPACE::TensorShapeProto* past_shape,
                       const ONNX_NAMESPACE::TensorShapeProto* logits_shape,
                       bool merged_past);

  AllocatorPtr allocator_;
  const SessionState* session_state_;
  const SessionState* subgraph_session_state_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;
  bool is_output_float16_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
