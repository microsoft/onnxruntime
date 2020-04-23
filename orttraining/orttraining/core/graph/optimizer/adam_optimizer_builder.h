// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/optimizer_builder.h"

namespace onnxruntime {
namespace training {

class AdamOptimizerBuilder final : public OptimizerBuilder {
 public:
  AdamOptimizerBuilder() : OptimizerBuilder(OpDef{"AdamOptimizer", kMSDomain, 1},
                                            {"alpha",
                                             "beta",
                                             "lambda",
                                             "epsilon",
                                             "do_bias_correction",
                                             "weight_decay_mode"}) {}

  virtual Status Build(
      const std::vector<ArgDef>& weight_argdefs,
      const std::vector<ArgDef>& gradient_argdefs,
      const ArgDef* gradient_norm_argdef,
      const ArgDef* gradient_norm_finite_argdef,
      const std::vector<OptimizerNodeConfig>& opt_configs,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ONNX_NAMESPACE::TensorProto>& new_external_initializers,
      std::vector<ArgDef>& output_weight_argdefs,
      std::vector<ArgDef>& output_gradient_argdefs) const override;

  virtual Status Build(
      const std::vector<ArgDef>& weight_argdefs,
      const std::vector<ArgDef>& gradient_argdefs,
      const ArgDef* gradient_norm_argdef,
      const ArgDef* gradient_norm_finite_argdef,
      const std::vector<OptimizerNodeConfig>& opt_configs,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ONNX_NAMESPACE::TensorProto>& new_external_initializers,
      std::vector<ArgDef>& output_weight_argdefs,
      std::vector<ArgDef>& output_gradient_argdefs,
      const bool enable_grad_clipping) const override;
};

}  // namespace training
}  // namespace onnxruntime
