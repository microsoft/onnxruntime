// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/graph/training/in_graph_training_optimizer.h"
#include "core/graph/training/gradient_builder_base.h"
#include "core/graph/training/graph_augmenter.h"

// TODO: solve the op version issue in the entire training framework
// Here is to reference GRADIENT_OP_VERSION temporarily for global version control.
#include "core/graph/training/gradient_op_schema.h"

using namespace std;
namespace onnxruntime {
namespace training {
namespace in_graph_optimizer {

class SGDBuilder : public OptimizerBuilder {
 public:
  SGDBuilder() : OptimizerBuilder("SGDOptimizer") {}

  Status Build(const std::vector<std::string>& weights,
               const std::vector<std::string>& gradients,
               const OptimizerInfo& opt_info,
               GraphAugmenter::GraphDefs& graph_defs) const override {
    ORT_RETURN_IF_NOT(opt_info.params_.size() == 1, "SGDOptimizer should have 1 parameter (learning rate)");

    // The type proto for learning rate.
    TypeProto& lr_type_proto = *graph_defs.CreateTypeProto();
    lr_type_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    lr_type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    // inputs : learning_rate + weights + grads
    vector<ArgDef> input_args;
    input_args.emplace_back(opt_info.params_[0], &lr_type_proto);

    for (const auto& weight : weights) {
      input_args.emplace_back(weight);
    }

    for (const auto& grad : gradients) {
      input_args.emplace_back(grad);
    }

    // outputs: new weights, also set as graph outputs
    vector<ArgDef> output_args;
    for (const auto& weight : weights) {
      string output_name = weight + "_SGD_out";
      output_args.emplace_back(output_name);
      graph_defs.AddGraphOutputs({output_name});
    }

    graph_defs.AddNodeDefs({NodeDef("SGDOptimizer", input_args, output_args)});
    return Status::OK();
  }
};

#define REGISTER_OPTIMIZER_BUILDER(op_name, optimizer_builder) \
  GetInstance().Register<optimizer_builder>(op_name);

// Register all optimerzer here.
void OptimizerBuilderRegistry::RegisterBuilders() {
  REGISTER_OPTIMIZER_BUILDER("SGDOptimizer", SGDBuilder);
}
}  // namespace in_graph_optimizer
}  // namespace training
}  // namespace onnxruntime
