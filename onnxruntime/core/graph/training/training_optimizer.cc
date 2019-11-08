// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/graph/training/training_optimizer.h"
#include "core/graph/training/gradient_builder_base.h"
#include "core/graph/training/graph_augmenter.h"

namespace onnxruntime {
namespace training {

Status SGDOptimizerBuilder::Build(
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* /* do_update_argdef */,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& external_inputs_including_initializers,
    std::vector<TensorProto>& /* new_external_initializers */,
    std::vector<ArgDef>& output_weight_argdefs) const {
  
  for (size_t i = 0; i < weight_argdefs.size(); ++i) {
    const std::string& weight_name = weight_argdefs[i].name;
    const TypeProto* const weight_type_proto = weight_argdefs[i].type_proto;

    std::vector<ArgDef> input_args(num_inputs_);
    input_args[0] = ArgDef(opt_configs[i].lr_feed_name);
    input_args[1] = weight_argdefs[i];
    input_args[2] = gradient_argdefs[i];

    std::vector<ArgDef> output_args(num_outputs_);
    output_args[0] = ArgDef(weight_name + "_SGD_out", weight_type_proto);  // output 0 new weights

    graph_defs.AddNodeDefs({NodeDef(OpType(),
                                    input_args,
                                    output_args,
                                    NodeAttributes(),
                                    OptimizerNodeName(weight_name))});

    for (auto &arg: input_args) {
      external_inputs_including_initializers.emplace_back(arg);
    }
    output_weight_argdefs[i] = output_args[0];
  }

  return Status::OK();
}

Status AdamOptimizerBuilder::Build(
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* do_update_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& external_inputs_including_initializers,
    std::vector<TensorProto>& new_external_initializers,
    std::vector<ArgDef>& output_weight_argdefs) const {
  std::vector<TensorProto> new_initializers{};
  for (size_t i = 0; i < weight_argdefs.size(); ++i) {
    const std::string& weight_name = weight_argdefs[i].name;
    const std::string& gradient_name = gradient_argdefs[i].name;
    const TypeProto* const weight_type_proto = weight_argdefs[i].type_proto;

    // The type proto initializer for Update Count
    const std::string update_count_string = "Update_Count_" + weight_name;  // per weight optimizer requires a per weight update count
    TensorProto uc_tensor_proto = CreateTensorProto<int64_t>(update_count_string, 1);
    // Add uc tensorproto as initializers
    new_initializers.emplace_back(uc_tensor_proto);

    int num_inputs = num_inputs_;
    int num_outputs = num_outputs_;

    // When mixed precision is enabled by using FP16 initializer, optimizer consumes fp32 weight tensor and its fp16 copy.
    // Thus, each optimizer will get one extra input and one extra output.
    if (opt_configs[i].fp16_weight_arg != nullptr) {
      num_inputs += 1;
      num_outputs += 1;
    }

    if (!opt_configs[i].loss_scale_input_name.empty()) {
      num_inputs += 1;
    }

    if (do_update_argdef) {
      num_inputs += 1;
    }

    std::vector<ArgDef> input_args(num_inputs);
    input_args[0] = ArgDef(opt_configs[i].lr_feed_name);
    input_args[1] = ArgDef(update_count_string);
    input_args[2] = weight_argdefs[i];
    input_args[3] = gradient_argdefs[i];

    std::vector<ArgDef> output_args(num_outputs);
    output_args[0] = ArgDef(weight_name + "_Adam_out", weight_type_proto);

    // Create the tensor proto for first and second moments of grad.
    int input_idx = 4;
    int output_idx = 1;

    // Get shape of weight tensor.
    std::vector<int64_t> weight_dims;
    ORT_RETURN_IF_NOT(
        weight_argdefs[i].type_proto &&
        weight_argdefs[i].type_proto->has_tensor_type() &&
        weight_argdefs[i].type_proto->tensor_type().has_shape());
    for (const auto& dim : weight_argdefs[i].type_proto->tensor_type().shape().dim()) {
      weight_dims.push_back(dim.dim_value());
    }

    // Add first- and second-order momentums to input list.
    const std::vector<std::string> moments_prefixes({"Moment_1_", "Moment_2_"});
    for (const auto& moments_prefix : moments_prefixes) {
      const std::string gradient_moment_name = moments_prefix + gradient_name;

      TensorProto moment_tensor_proto;
      TypeProto* moment_type_proto = graph_defs.CopyTypeProto(weight_argdefs[i]);
      if (opt_configs[i].use_fp16_moments) {
        moment_tensor_proto = CreateTensorProto<MLFloat16>(gradient_moment_name, MLFloat16(math::floatToHalf(0.f)), weight_dims);
        moment_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16);
      } else {
        moment_tensor_proto = CreateTensorProto<float>(gradient_moment_name, 0.f, weight_dims);
      }

      new_initializers.emplace_back(moment_tensor_proto);

      input_args[input_idx++] = ArgDef(gradient_moment_name, moment_type_proto);
      output_args[output_idx++] = ArgDef(gradient_moment_name + "_Out", moment_type_proto);
    }

    TypeProto* step_type_proto = graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_INT64);
    output_args[3] = ArgDef(gradient_name + "_Step_Out", step_type_proto);

    if (opt_configs[i].fp16_weight_arg != nullptr) {
      input_args[6] = ArgDef(opt_configs[i].fp16_weight_arg->Name(), opt_configs[i].fp16_weight_arg->TypeAsProto());
      std::string output_name = opt_configs[i].fp16_weight_arg->Name() + "_Adam_out";
      output_args[4] = ArgDef(output_name, opt_configs[i].fp16_weight_arg->TypeAsProto());
    }

    if (!opt_configs[i].loss_scale_input_name.empty()) {
      input_args[7] = ArgDef(opt_configs[i].loss_scale_input_name, graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    }

    if (do_update_argdef) {
      input_args[8] = *do_update_argdef;
    }

    graph_defs.AddNodeDefs({NodeDef(OpType(),
                                    input_args,
                                    output_args,
                                    BuildAttributeProto(opt_configs[i]),
                                    OptimizerNodeName(weight_name))});

    for (auto& arg : input_args) {
      external_inputs_including_initializers.emplace_back(arg);
    }
    output_weight_argdefs[i] = output_args[0];
  }
  new_external_initializers = std::move(new_initializers);
  return Status::OK();
}

// external_inputs_including_initializers are the input NoeArg's
// required by the created optimizer node. This parameter is used
// to return the optimizer-specific input NodeArgs created in this function
// to the enclosing scope.
// output_weight_argdefs is used to pass the updated-weight output
// NodeArgs created in this function to the enclosing scope.
Status LambOptimizerBuilder::Build(
    const std::vector<ArgDef>& weight_argdefs,
    const std::vector<ArgDef>& gradient_argdefs,
    const ArgDef* do_update_argdef,
    const std::vector<OptimizerNodeConfig>& opt_configs,
    GraphAugmenter::GraphDefs& graph_defs,
    std::vector<ArgDef>& external_inputs_including_initializers,
    std::vector<TensorProto>& new_external_initializers,
    std::vector<ArgDef>& output_weight_argdefs) const {
  ORT_ENFORCE(weight_argdefs.size() <= size_t(1024),
    "The current LambOptimizer can only update up to 1024 weight tensors, but",
    "the actual number of weight tensors is ", weight_argdefs.size());
  // The variable stores optimizer's states such as momentums. Those
  // state don't appear in the inference graph, so
  // we add them as initializers.
  std::vector<TensorProto> new_initializers{};

  // Lamb optimizer node's inputs and outputs.
  std::vector<ArgDef> input_argdefs;
  std::vector<ArgDef> output_argdefs;

  // Loss scale ArgDef.
  if (!opt_configs[0].loss_scale_input_name.empty()) {
    input_argdefs.emplace_back(ArgDef(opt_configs[0].loss_scale_input_name, graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT)));
  } else {
    input_argdefs.emplace_back(ArgDef());
  }

  // Update signal ArgDef.
  if (do_update_argdef) {
    input_argdefs.push_back(*do_update_argdef);
  } else {
    input_argdefs.emplace_back(ArgDef());
  }

  // Learning rate ArgDef.
  input_argdefs.emplace_back(ArgDef(opt_configs[0].lr_feed_name));

  // Lamb optimizer's attributes.
  std::vector<float> alpha;
  std::vector<float> beta;
  std::vector<float> lambda;
  std::vector<float> epsilon;
  std::vector<float> threshold;

  // Each iteration handles the associated inputs and outputs of a weight tensor.
  // Associated inputs: [w, g, m1, m2, w_fp16].
  // Associated outputs: [w_new, m1_new, m2_new, w_fp16_new].
  for (size_t i = 0; i < weight_argdefs.size(); ++i) {
    const std::string& weight_name = weight_argdefs[i].name;
    const std::string& weight_new_name = weight_argdefs[i].name + "_Lamb_out";
    const std::string& gradient_name = GradientBuilderBase::GradientName(weight_name);

    const auto& attrs = opt_configs[i].attributes;

    {
      auto alpha_iter = attrs.find("alpha");
      if (alpha_iter != attrs.end())
        alpha.emplace_back(alpha_iter->second);
      else
        alpha.emplace_back(0.9f);
    }

    {
      auto beta_iter = attrs.find("beta");
      if (beta_iter != attrs.end())
        beta.emplace_back(beta_iter->second);
      else
        beta.emplace_back(0.999f);
    }
   
    {
      auto lambda_iter = attrs.find("lambda");
      if (lambda_iter != attrs.end())
        lambda.emplace_back(lambda_iter->second);
      else
        lambda.emplace_back(0.0f);
    }

    {
      auto epsilon_iter = attrs.find("epsilon");
      if (epsilon_iter != attrs.end())
        epsilon.emplace_back(epsilon_iter->second);
      else
        epsilon.emplace_back(1e-6f);
    }
    
    {
      auto threshold_iter = attrs.find("threshold");
      if (threshold_iter != attrs.end())
        threshold.emplace_back(threshold_iter->second);
      else
        threshold.emplace_back(1.0f);
    }

    // Extract weight's type and shape information.
    const TypeProto* const weight_type_proto = weight_argdefs[i].type_proto;
    std::vector<int64_t> weight_dims;
    ORT_RETURN_IF_NOT(
        weight_argdefs[i].type_proto &&
        weight_argdefs[i].type_proto->has_tensor_type() &&
        weight_argdefs[i].type_proto->tensor_type().has_shape());
    for (const auto& dim : weight_argdefs[i].type_proto->tensor_type().shape().dim()) {
      weight_dims.push_back(dim.dim_value());
    }

    // w & g & w_new
    input_argdefs.push_back(weight_argdefs[i]);
    input_argdefs.push_back(gradient_argdefs[i]);
    output_argdefs.emplace_back(ArgDef(weight_new_name, weight_type_proto));

    // m1 & m2 & m1_new & m2_new
    const std::vector<std::string> moments_prefixes({"Moment_1_", "Moment_2_"});
    for (const auto& moment_prefix : moments_prefixes) {
      const std::string gradient_moment_name = moment_prefix + gradient_name;

      // Construct type of momentum tensor.
      TensorProto moment_tensor_proto;
      TypeProto* moment_type_proto = graph_defs.CopyTypeProto(weight_argdefs[i]);
      if (opt_configs[i].use_fp16_moments) {
        moment_tensor_proto = CreateTensorProto<MLFloat16>(gradient_moment_name, MLFloat16(math::floatToHalf(0.f)), weight_dims);
        moment_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16);
      } else {
        moment_tensor_proto = CreateTensorProto<float>(gradient_moment_name, 0.f, weight_dims);
      }

      // Store momentum tensor to initializer list.
      new_external_initializers.emplace_back(std::move(moment_tensor_proto));

      // Add momentums to the input and output list of the Lamb node.
      input_argdefs.emplace_back(ArgDef(gradient_moment_name, moment_type_proto));
      output_argdefs.emplace_back(ArgDef(gradient_moment_name + "_Out", moment_type_proto));
    }

    // w_fp16 & w_fp16_new
    if (opt_configs[i].fp16_weight_arg != nullptr) {
      input_argdefs.emplace_back(ArgDef(
        opt_configs[i].fp16_weight_arg->Name(),
        opt_configs[i].fp16_weight_arg->TypeAsProto()));
      output_argdefs.emplace_back(ArgDef(
        opt_configs[i].fp16_weight_arg->Name() + "_Lamb_out",
        opt_configs[i].fp16_weight_arg->TypeAsProto()));
    } else {
      input_argdefs.emplace_back(ArgDef());
      output_argdefs.emplace_back(ArgDef());
    }

    for (auto& arg : input_argdefs) {
      external_inputs_including_initializers.push_back(arg);
    }

    output_weight_argdefs[i] = ArgDef(weight_new_name, weight_type_proto);
  }

  std::vector<AttributeProto> attribute_protos;
  attribute_protos.emplace_back(MakeAttribute("alpha", alpha));
  attribute_protos.emplace_back(MakeAttribute("beta", beta));
  attribute_protos.emplace_back(MakeAttribute("lambda", lambda));
  attribute_protos.emplace_back(MakeAttribute("epsilon", epsilon));
  attribute_protos.emplace_back(MakeAttribute("threshold", threshold));

  graph_defs.AddNodeDefs({NodeDef(OpType(),
                                  input_argdefs,
                                  output_argdefs,
                                  attribute_protos,
                                  OptimizerNodeName(OpType()))});

  return Status::OK();
}

#define REGISTER_OPTIMIZER_BUILDER(op_name, optimizer_builder) \
  GetInstance().Register<optimizer_builder>(op_name);

// Register all optimizers here.
void OptimizerBuilderRegistry::RegisterBuilders() {
  REGISTER_OPTIMIZER_BUILDER("SGDOptimizer", SGDOptimizerBuilder);
  REGISTER_OPTIMIZER_BUILDER("AdamOptimizer", AdamOptimizerBuilder);
  REGISTER_OPTIMIZER_BUILDER("LambOptimizer", LambOptimizerBuilder);
}

}  // namespace training
}  // namespace onnxruntime
