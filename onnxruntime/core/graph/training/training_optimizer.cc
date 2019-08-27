// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/graph/training/training_optimizer.h"
#include "core/graph/training/gradient_builder_base.h"
#include "core/graph/training/graph_augmenter.h"

// TODO: solve the op version issue in the entire training framework
// Here is to reference GRADIENT_OP_VERSION temporarily for global version control.
#include "core/graph/training/gradient_op_schema.h"

namespace onnxruntime {
namespace training {

#ifndef USE_HOROVOD
std::string BuildAllReduceNode(const std::string& /*gradient*/, GraphAugmenter::GraphDefs& /*graph_defs*/) {
  ORT_NOT_IMPLEMENTED("Distributed training is not supported, as Horovod is not enabled in this build.");
}
#else
std::string BuildAllReduceNode(const std::string& gradient, GraphAugmenter::GraphDefs& graph_defs) {
  const std::string allreduce_output = gradient + "_AllReduce_Out";
  graph_defs.AddNodeDefs({NodeDef("HorovodAllReduceOp",
                                  {ArgDef(gradient)},
                                  {ArgDef(allreduce_output)},
                                  NodeAttributes(),
                                  gradient)});
  return allreduce_output;
}
#endif

class SGDBuilder : public OptimizerBuilder {
 public:
  SGDBuilder() : OptimizerBuilder("SGDOptimizer") {}

  Status Build(const NodeArg* weight_arg,
               const OptimizerInfo& opt_info,
               GraphAugmenter::GraphDefs& graph_defs) const override {
    const std::string gradient = GradientBuilderBase::GradientName(weight_arg->Name());

    // Initialized tensor for Learning Rate
    TensorProto lr_tensor_proto = CreateTensorProto<float>(learning_rate_string_, opt_info.learning_rate);
    graph_defs.AddInitializers({lr_tensor_proto});

    std::vector<ArgDef> input_args;
    input_args.emplace_back(learning_rate_string_);                          // input 0 learning_rate
    input_args.emplace_back(weight_arg->Name(), weight_arg->TypeAsProto());  // input 1 weight

    std::string processed_gradient = gradient;
    if (opt_info.world_size > 1) {
      processed_gradient = BuildAllReduceNode(gradient, graph_defs);
    }

    input_args.emplace_back(processed_gradient);  // input 2:  allreduce output or gradient

    ArgDef output_arg(weight_arg->Name() + "_SGD_out", weight_arg->TypeAsProto());  // output 0 new weights

    graph_defs.AddNodeDefs({NodeDef("SGDOptimizer",
                                    input_args,
                                    {output_arg},
                                    NodeAttributes(),
                                    "SGDOptimizer_" + weight_arg->Name())});
    return Status::OK();
  }
};

class AdamOptimizerBuilder : public OptimizerBuilder {
 public:
  AdamOptimizerBuilder() : OptimizerBuilder("AdamOptimizer") {}

  Status Build(const NodeArg* weight_arg,
               const OptimizerInfo& opt_info,
               GraphAugmenter::GraphDefs& graph_defs) const override {
    const std::string gradient = GradientBuilderBase::GradientName(weight_arg->Name());

    // Initialized tensor for Learning Rate
    TensorProto lr_tensor_proto = CreateTensorProto<float>(learning_rate_string_, opt_info.learning_rate);

    // The type proto initializer for Update Count
    std::string update_count_string = "Update_Count_" + weight_arg->Name();  // per weight optimizer requires a per weight update count
    TensorProto uc_tensor_proto = CreateTensorProto<int64_t>(update_count_string, 1);

    // Add lr and uc tensorproto as initializers
    graph_defs.AddInitializers({lr_tensor_proto, uc_tensor_proto});

    std::vector<ArgDef> input_args;
    input_args.emplace_back(learning_rate_string_);                          // input 0 learning_rate
    input_args.emplace_back(update_count_string);                            // input 1 update_count
    input_args.emplace_back(weight_arg->Name(), weight_arg->TypeAsProto());  // input 2 weights

    std::string processed_gradient = gradient;
    if (opt_info.world_size > 1) {
      processed_gradient = BuildAllReduceNode(gradient, graph_defs);
    }

    input_args.emplace_back(processed_gradient);  // input 3 gradient or allreduce output

    // The tensor proto for first and second moments of grad
    std::vector<std::string> moments_strings({"Moment_1_", "Moment_2_"});
    for (auto moment_string : moments_strings) {
      std::string gradient_moment_name = moment_string + gradient;
      std::vector<int64_t> dims;
      for (auto dim : weight_arg->Shape()->dim()) {
        dims.push_back(dim.dim_value());
      }
      TensorProto moment_tensor_proto = CreateTensorProto<float>(gradient_moment_name, 0.f, dims);
      graph_defs.AddInitializers({moment_tensor_proto});
      input_args.emplace_back(gradient_moment_name);  // input 4 moment_1, input 5 moment_2
    }

    std::vector<ArgDef> output_args;
    output_args.emplace_back(weight_arg->Name() + "_Adam_out", weight_arg->TypeAsProto());  // output 0 new weights
    output_args.emplace_back(gradient + "_Moment_1_Out", weight_arg->TypeAsProto());        // output 1 moment 1
    output_args.emplace_back(gradient + "_Moment_2_Out", weight_arg->TypeAsProto());        // output 2 moment 2

    TypeProto* type_proto = graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_INT64);
    output_args.emplace_back(gradient + "_Step_Out", type_proto);  //output 3 step count

    std::vector<std::string> attr_names{"alpha", "beta", "lambda", "epsilon"};
    std::vector<AttributeProto> attr;
    for (auto name : attr_names) {
      attr.push_back(MakeAttribute(name, opt_info.attributes_.at(name)));
    }

    graph_defs.AddNodeDefs({NodeDef("AdamOptimizer",
                                    input_args,
                                    output_args,
                                    attr,
                                    "AdamOptimizer_" + weight_arg->Name())});
    return Status::OK();
  }
};

#define REGISTER_OPTIMIZER_BUILDER(op_name, optimizer_builder) \
  GetInstance().Register<optimizer_builder>(op_name);

// Register all optimizers here.
void OptimizerBuilderRegistry::RegisterBuilders() {
  REGISTER_OPTIMIZER_BUILDER("SGDOptimizer", SGDBuilder);
  REGISTER_OPTIMIZER_BUILDER("AdamOptimizer", AdamOptimizerBuilder);
}
}  // namespace training
}  // namespace onnxruntime
