
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer_config.h"

namespace onnxruntime {

namespace training {

struct TrainingGraphTransformerConfiguration : public GraphTransformerConfiguration {
  bool enable_gelu_approximation{false};

  // Enable recompute of attention dropout to save memory
  bool attn_dropout_recompute{false};
  // Enable recompute of Gelu activation output to save memory
  bool gelu_recompute{false};
  // Enable recompute of transformer layer ouput to save memory
  bool transformer_layer_recompute{false};
  // Number of layers to apply recompute
  int number_recompute_layers{0};

  // Enable compute optimizer.
  bool enable_compute_optimizer{false};

  // Enable embedding sparsity compute optimization for the input names in the below list.
  std::vector<std::string> sparse_embedding_input_names;

  // Enable label sparsity compute optimization for the input names in the below list.
  std::vector<std::string> sparse_label_input_names;
};

}  // namespace training
}  // namespace onnxruntime
