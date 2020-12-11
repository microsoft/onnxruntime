// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/optimizer/sgd_optimizer_builder.h"
#include "orttraining/core/graph/graph_augmenter.h"

namespace onnxruntime {
namespace training {
Status SGDOptimizerBuilder::Build(
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* gradient_norm_argdef,
    const ArgDef* gradient_norm_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ONNX_NAMESPACE::TensorProto>& new_external_initializers,
    std::vector<ArgDef>& output_weight_argdefs,
    std::vector<ArgDef>& output_gradient_argdefs) const {
  return Build(weight_argdefs, gradient_argdefs,
        gradient_norm_argdef, gradient_norm_finite_argdef,
        opt_configs, graph_defs,
        new_external_initializers, output_weight_argdefs,
        output_gradient_argdefs,
        // gradient clipping is disabled by default for SGD.
        false /*enable_grad_clipping*/);
}

Status SGDOptimizerBuilder::Build(
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* /* gradient_norm_argdef */,
    const ArgDef* /* gradient_norm_finite_argdef */,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<TensorProto>& /* new_external_initializers */,
    std::vector<ArgDef>& output_weight_argdefs,
    std::vector<ArgDef>& output_gradient_argdefs,
    bool /* enable_grad_clipping */) const {

  for (size_t i = 0; i < weight_argdefs.size(); ++i) {
    const std::string& weight_name = weight_argdefs[i].name;
    const std::string& gradient_name = gradient_argdefs[i].name;
    const TypeProto* const weight_type_proto = weight_argdefs[i].type_proto;
    const TypeProto* const gradient_type_proto = gradient_argdefs[i].type_proto;

    // Return either the input gradient/weight or updated gradient/weight.
    ArgDef output_gradient_argdef = gradient_argdefs[i];
    ArgDef output_weight_argdef = weight_argdefs[i];

    // In distributed training, some weights may not be updated by all ranks.
    if (opt_configs[i].enabled) {
      std::vector<ArgDef> input_args;
      input_args.push_back(ArgDef(opt_configs[i].lr_feed_name, CreateLearningRateTypeProto(graph_defs)));
      graph_defs.AddGraphInputs({opt_configs[i].lr_feed_name});
      input_args.push_back(weight_argdefs[i]);
      input_args.push_back(gradient_argdefs[i]);

      std::vector<ArgDef> output_args;
      if (opt_configs[i].update_weight) {
        output_weight_argdef = ArgDef(weight_name + "_SGD_out", weight_type_proto);
        output_args.push_back(output_weight_argdef);  // w_new
        output_args.push_back(ArgDef());  // g_new
      } else {
        output_gradient_argdef = ArgDef(gradient_name + "_SGD_out", gradient_type_proto);
        output_args.push_back(ArgDef());  // w_new
        output_args.push_back(output_gradient_argdef);  // g_new
      }

      graph_defs.AddNodeDefs({NodeDef(OpDefinition(),
                                      input_args,
                                      output_args,
                                      NodeAttributes(),
                                      OptimizerNodeName(weight_name))});
    }

    output_weight_argdefs.push_back(output_weight_argdef);
    output_gradient_argdefs.push_back(output_gradient_argdef);
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
