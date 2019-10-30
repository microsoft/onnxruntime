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

Status LambOptimizerBuilder::Build(
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
    const std::string& gradient_name = GradientBuilderBase::GradientName(weight_name);

    const TypeProto* const weight_type_proto = weight_argdefs[i].type_proto;

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
    input_args[1] = weight_argdefs[i];
    input_args[2] = gradient_argdefs[i];

    std::vector<ArgDef> output_args(num_outputs);
    output_args[0] = ArgDef(weight_name + "_Lamb_out", weight_type_proto);

    // The tensor proto for first and second moments of grad
    int input_idx = 3;
    int output_idx = 1;

    std::vector<int64_t> weight_dims;
    ORT_RETURN_IF_NOT(
        weight_argdefs[i].type_proto &&
        weight_argdefs[i].type_proto->has_tensor_type() &&
        weight_argdefs[i].type_proto->tensor_type().has_shape());
    for (const auto& dim : weight_argdefs[i].type_proto->tensor_type().shape().dim()) {
      weight_dims.push_back(dim.dim_value());
    }

    const std::vector<std::string> moments_prefixes({"Moment_1_", "Moment_2_"});
    for (const auto& moment_prefix : moments_prefixes) {
      const std::string gradient_moment_name = moment_prefix + gradient_name;

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

    if (opt_configs[i].fp16_weight_arg != nullptr) {
      input_args[5] = ArgDef(opt_configs[i].fp16_weight_arg->Name(), opt_configs[i].fp16_weight_arg->TypeAsProto());
      std::string output_name = opt_configs[i].fp16_weight_arg->Name() + "_Lamb_out";
      output_args[3] = ArgDef(output_name, opt_configs[i].fp16_weight_arg->TypeAsProto());
    }

    if (!opt_configs[i].loss_scale_input_name.empty()) {
      input_args[6] = ArgDef(opt_configs[i].loss_scale_input_name, graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    }

    if (do_update_argdef) {
      input_args[7] = *do_update_argdef;
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
