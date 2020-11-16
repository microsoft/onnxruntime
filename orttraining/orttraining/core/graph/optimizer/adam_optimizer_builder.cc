// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/optimizer/adam_optimizer_builder.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "core/util/math.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace training {
Status AdamOptimizerBuilder::Build(
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
               // gradient clipping is disabled by default for Adam.
               false /*enable_grad_clipping*/,
               {} /* default shared initital states*/);
}

Status AdamOptimizerBuilder::Build(
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* gradient_norm_argdef,
    const ArgDef* gradient_norm_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ONNX_NAMESPACE::TensorProto>& new_external_initializers,
    std::vector<ArgDef>& output_weight_argdefs,
    std::vector<ArgDef>& output_gradient_argdefs,
    bool enable_grad_clipping) const {
  return Build(weight_argdefs, gradient_argdefs,
               gradient_norm_argdef, gradient_norm_finite_argdef,
               opt_configs, graph_defs,
               new_external_initializers, output_weight_argdefs,
               output_gradient_argdefs, enable_grad_clipping,
               {} /* default shared initital states*/);
}

Status AdamOptimizerBuilder::Build(
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* gradient_norm_argdef,
    const ArgDef* gradient_norm_finite_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<TensorProto>& new_external_initializers,
    std::vector<ArgDef>& output_weight_argdefs,
    std::vector<ArgDef>& output_gradient_argdefs,
    bool enable_grad_clipping,
    const NameMLValMap) const {
  for (size_t i = 0; i < weight_argdefs.size(); ++i) {
    const std::string& weight_name = weight_argdefs[i].name;
    const std::string& gradient_name = gradient_argdefs[i].name;
    const TypeProto* const weight_type_proto = weight_argdefs[i].type_proto;
    const TypeProto* const gradient_type_proto = gradient_argdefs[i].type_proto;

    // Return either the input gradient/weight/mixed-precision-weight or updated gradient/weight/mixed-precision-weight.
    ArgDef output_gradient_argdef = gradient_argdefs[i];
    ArgDef output_weight_argdef = weight_argdefs[i];
    if (opt_configs[i].mixed_precision_weight_arg != nullptr)
      output_weight_argdef = ArgDef(opt_configs[i].mixed_precision_weight_arg->Name(), opt_configs[i].mixed_precision_weight_arg->TypeAsProto());

    // In distributed training, some weights may not be updated by all ranks.
    if (opt_configs[i].enabled) {
      // The type proto initializer for Update Count
      std::string uc_prefix = "Update_Count";
      const std::string update_count_string = uc_prefix + "_" + weight_name;  // per weight optimizer requires a per weight update count
      TensorProto uc_tensor_proto;

      // Update 'Update_Count' initializer with init value
      const auto initial_states = opt_configs[i].initial_states;
      if (initial_states.find(uc_prefix) != initial_states.end()) {
        std::cout << "Adam: Updating Update_Count...\n";
        const auto& initial_state_it = initial_states.find(uc_prefix);
        const auto& init_tensor = initial_state_it->second.Get<Tensor>();
        ORT_ENFORCE(IsMatchingTypeAndShape(init_tensor, ONNX_NAMESPACE::TensorProto_DataType_INT64, {1}).IsOK());
        uc_tensor_proto = utils::TensorToTensorProto(init_tensor, update_count_string);
      } else {
        uc_tensor_proto = CreateTensorProto<int64_t>(update_count_string, 1);
      }

      // Add uc tensorproto as initializers
      new_external_initializers.emplace_back(uc_tensor_proto);

      std::vector<ArgDef> input_args;
      input_args.push_back(ArgDef(opt_configs[i].lr_feed_name, CreateLearningRateTypeProto(graph_defs)));
      graph_defs.AddGraphInputs({opt_configs[i].lr_feed_name});
      input_args.push_back(ArgDef(update_count_string));
      input_args.push_back(weight_argdefs[i]);
      input_args.push_back(gradient_argdefs[i]);

      std::vector<ArgDef> output_args;

      TypeProto* step_type_proto = graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_INT64);
      output_args.push_back(ArgDef(gradient_name + "_Step_Out", step_type_proto));

      // Get shape of weight tensor.
      std::vector<int64_t> weight_dims;
      ORT_RETURN_IF_NOT(
          weight_argdefs[i].type_proto &&
          weight_argdefs[i].type_proto->has_tensor_type() &&
          weight_argdefs[i].type_proto->tensor_type().has_shape());
      for (const auto& dim : weight_argdefs[i].type_proto->tensor_type().shape().dim()) {
        weight_dims.push_back(dim.dim_value());
      }

      auto element_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT;
      if (opt_configs[i].use_mixed_precision_moments) {
        element_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;
      }

      // Add first- and second-order momentums to input list.
      const std::vector<std::string> moments_prefixes({"Moment_1", "Moment_2"});
      for (const auto& moments_prefix : moments_prefixes) {
        const std::string gradient_moment_name = moments_prefix + "_" + weight_name;

        TensorProto moment_tensor_proto;
        TypeProto* moment_type_proto = graph_defs.CopyTypeProto(weight_argdefs[i]);

        // Update moment initializer with init value
        if (initial_states.find(moments_prefix) != initial_states.end()) {
          //update moment_tensor_proto
          std::cout << "Adam: Updating moment_tensor_proto...\n";
          const auto& initial_state_it = initial_states.find(moments_prefix);
          const auto& init_tensor = initial_state_it->second.Get<Tensor>();

          //TODO: need to support float -> float16 and float16-> float conversion
          ORT_ENFORCE(IsMatchingTypeAndShape(init_tensor, element_type, weight_dims).IsOK());
          moment_tensor_proto = utils::TensorToTensorProto(init_tensor, gradient_moment_name);
        } else if (opt_configs[i].use_mixed_precision_moments) {
          moment_tensor_proto = CreateTensorProto<MLFloat16>(gradient_moment_name, MLFloat16(math::floatToHalf(0.f)), weight_dims);
          moment_type_proto->mutable_tensor_type()->set_elem_type(element_type);
        } else {
          moment_tensor_proto = CreateTensorProto<float>(gradient_moment_name, 0.f, weight_dims);
        }

        new_external_initializers.emplace_back(moment_tensor_proto);

        input_args.push_back(ArgDef(gradient_moment_name, moment_type_proto));
        output_args.push_back(ArgDef(gradient_moment_name + "_Out", moment_type_proto));
      }

      // Output either w_new or g_new based on config.
      if (opt_configs[i].update_weight) {
        output_weight_argdef = ArgDef(weight_name + "_Adam_out", weight_type_proto);
        output_args.push_back(output_weight_argdef);  // w_new
        output_args.push_back(ArgDef());              // g_new
      } else {
        output_gradient_argdef = ArgDef(gradient_name + "_Adam_out", gradient_type_proto);
        output_args.push_back(ArgDef());                // w_new
        output_args.push_back(output_gradient_argdef);  // g_new
      }

      if (opt_configs[i].update_weight && opt_configs[i].mixed_precision_weight_arg != nullptr) {
        input_args.push_back(ArgDef(opt_configs[i].mixed_precision_weight_arg->Name(), opt_configs[i].mixed_precision_weight_arg->TypeAsProto()));
        std::string output_name = opt_configs[i].mixed_precision_weight_arg->Name() + "_Adam_out";
        output_weight_argdef = ArgDef(output_name, opt_configs[i].mixed_precision_weight_arg->TypeAsProto());
        output_args.push_back(output_weight_argdef);
      } else {
        input_args.push_back(ArgDef());
        output_args.push_back(ArgDef());
      }
      if (!opt_configs[i].loss_scale_input_name.empty()) {
        input_args.emplace_back(ArgDef(opt_configs[i].loss_scale_input_name, graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT)));
      } else {
        input_args.emplace_back(ArgDef());
      }

      if (gradient_norm_argdef && enable_grad_clipping) {
        input_args.push_back(*gradient_norm_argdef);
      } else if (gradient_norm_argdef == nullptr && enable_grad_clipping) {
        ORT_THROW("Gradient clipping is enabled but gradient norm is not given.");
      } else {
        input_args.push_back(ArgDef());
      }

      if (gradient_norm_finite_argdef) {
        input_args.push_back(*gradient_norm_finite_argdef);
      } else {
        input_args.push_back(ArgDef());
      }

      graph_defs.AddNodeDefs({NodeDef(OpDefinition(),
                                      input_args,
                                      output_args,
                                      BuildAttributeProto(opt_configs[i]),
                                      OptimizerNodeName(weight_name))});
    }

    output_weight_argdefs.push_back(output_weight_argdef);
    output_gradient_argdefs.push_back(output_gradient_argdef);
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
