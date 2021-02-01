// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/optimizer_builder.h"

namespace onnxruntime {
namespace training {

class LambOptimizerBuilder final : public OptimizerBuilder {
 public:
  LambOptimizerBuilder() : OptimizerBuilder(OpDef{"LambOptimizer", kMSDomain, 1},
                                            {"alpha",
                                             "beta",
                                             "lambda",
                                             "epsilon",
                                             "max_norm_clip",
                                             "ratio_min",
                                             "ratio_max",
                                             "do_bias_correction"}) {}

  virtual Status Build(
      const OptimizerBuilderConfig& config,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ONNX_NAMESPACE::TensorProto>& new_external_initializers,
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
      std::vector<ArgDef>& output_weight_argdefs,
      std::vector<ArgDef>& output_gradient_argdefs) const override;
};

}  // namespace training
}  // namespace onnxruntime
