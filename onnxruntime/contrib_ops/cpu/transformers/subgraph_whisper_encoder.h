// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/subgraph_base.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// A class for T5 encoder subgraph inputs and outputs preparation.
class WhisperEncoderSubgraph : public T5EncoderSubgraph {
 public:
  WhisperEncoderSubgraph(
      const onnxruntime::Node& node_in,
      const std::string& attribute_name,
      const GraphViewer& subgraph_in) : Subgraph(node_in, attribute_name, subgraph_in) {
    first_present_output_index_ = 2;
  }
}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
