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

class SGDBuilder : public OptimizerBuilder {
 public:
  SGDBuilder() : OptimizerBuilder("SGDOptimizer") {}

  Status Build(const std::vector<const NodeArg*> weight_args,
               const OptimizerInfo& opt_info,
               GraphAugmenter::GraphDefs& graph_defs) const override {
    std::vector<std::string> gradients(weight_args.size());
    for (size_t i = 0; i < weight_args.size(); ++i) {
      gradients[i] = GradientBuilderBase::GradientName(weight_args[i]->Name());
    }

    // Initialized tensor for Learning Rate
    TensorProto lr_tensor_proto = CreateTensorProto<float>(learning_rate_string_, opt_info.learning_rate);
    graph_defs.AddInitializers({lr_tensor_proto});

    // inputs : learning_rate + weights + grads
    vector<ArgDef> input_args;
    input_args.emplace_back(learning_rate_string_);

    for (const auto& weight_arg : weight_args) {
      input_args.emplace_back(weight_arg->Name(), weight_arg->TypeAsProto());
    }

#ifdef USE_HOROVOD
    std::vector<ArgDef> agg_grads;
    if (opt_info.world_size > 1) {
      BuildAllReduceNode(gradients, graph_defs, agg_grads);
      for (const auto& grad : agg_grads) {
        input_args.emplace_back(grad);
      }
    } else {
      for (const auto& grad : gradients) {
        input_args.emplace_back(grad);
      }
    }
#else
    for (size_t i = 0; i < weight_args.size(); ++i) {
      input_args.emplace_back(gradients[i], weight_args[i]->TypeAsProto());
    }
#endif

    // outputs: new weights, also set as graph outputs
    vector<ArgDef> output_args;
    for (const auto& weight_arg : weight_args) {
      string output_name = weight_arg->Name() + "_SGD_out";
      output_args.emplace_back(output_name, weight_arg->TypeAsProto());
      graph_defs.AddGraphOutputs({output_name});
    }

    graph_defs.AddNodeDefs({NodeDef("SGDOptimizer",
                                    input_args,
                                    output_args,
                                    NodeAttributes(),
                                    "SGDOptimizer_" + weight_args[0]->Name())});
    return Status::OK();
  }
};

class AdamOptimizerBuilder : public OptimizerBuilder {
 public:
  AdamOptimizerBuilder() : OptimizerBuilder("AdamOptimizer") {}

  Status Build(const std::vector<const NodeArg*> weight_args,
               const OptimizerInfo& opt_info,
               GraphAugmenter::GraphDefs& graph_defs) const override {
    std::vector<std::string> gradients(weight_args.size());
    for (size_t i = 0; i < weight_args.size(); ++i) {
      gradients[i] = GradientBuilderBase::GradientName(weight_args[i]->Name());
    }

    // Initialized tensor for Learning Rate
    TensorProto lr_tensor_proto = CreateTensorProto<float>(learning_rate_string_, opt_info.learning_rate);

    // The type proto initializer for Update Count
    std::string update_count_string = "Update_Count" + ((weight_args.size() > 1) ? "" : "_" + weight_args[0]->Name());  // per weight optimizer requires a per weight update count
    TensorProto uc_tensor_proto = CreateTensorProto<int64_t>(update_count_string, 1);

    // Add lr and uc tensorproto as initializers
    graph_defs.AddInitializers({lr_tensor_proto, uc_tensor_proto});

    // inputs :learning_rate + update_count + weights + gradients + first moment + second moment
    vector<ArgDef> input_args;
    input_args.emplace_back(learning_rate_string_);
    input_args.emplace_back(update_count_string);

    for (const auto& weight_arg : weight_args) {
      input_args.emplace_back(weight_arg->Name(), weight_arg->TypeAsProto());
    }

#ifdef USE_HOROVOD
    std::vector<ArgDef> agg_grads;
    if (opt_info.world_size > 1) {
      BuildAllReduceNode(gradients, graph_defs, agg_grads);
      for (const auto& grad : agg_grads) {
        input_args.emplace_back(grad);
      }
    } else {
      for (const auto& grad : gradients) {
        input_args.emplace_back(grad);
      }
    }
#else
    for (size_t i = 0; i < gradients.size(); ++i) {
      input_args.emplace_back(gradients[i], weight_args[i]->TypeAsProto());
    }
#endif

    // The tensor proto for first and second moments of grad
    vector<string> moments_strings({"Moment_1_", "Moment_2_"});
    for (auto moment_string : moments_strings) {
      for (size_t i = 0; i < gradients.size(); i++) {
        std::string gradient_moment_name = moment_string + gradients[i];
        std::vector<int64_t> dims;
        for (auto dim : weight_args[i]->Shape()->dim()) {
          dims.push_back(dim.dim_value());
        }
        TensorProto moment_tensor_proto = CreateTensorProto<float>(gradient_moment_name, 0.f, dims);
        graph_defs.AddInitializers({moment_tensor_proto});
        input_args.emplace_back(gradient_moment_name);
      }
    }

    std::vector<string> attr_names{"alpha", "beta", "lambda", "epsilon"};
    std::vector<AttributeProto> attr;

    for (auto name : attr_names) {
      attr.push_back(MakeAttribute(name, opt_info.attributes_.at(name)));
    }

    // outputs: new weights, also set as graph outputs - This is not used currently
    vector<ArgDef> output_args;
    for (const auto& weight_arg : weight_args) {
      string output_name = weight_arg->Name() + "_Adam_out";
      output_args.emplace_back(output_name, weight_arg->TypeAsProto());
      graph_defs.AddGraphOutputs({output_name});
    }

    for (size_t i = 0; i < gradients.size(); ++i) {
      string output_name = gradients[i] + "_Moment_1_Out";
      output_args.emplace_back(output_name, weight_args[i]->TypeAsProto());
      graph_defs.AddGraphOutputs({output_name});
    }

    for (size_t i = 0; i < gradients.size(); ++i) {
      string output_name = gradients[i] + "_Moment_2_Out";
      output_args.emplace_back(output_name, weight_args[i]->TypeAsProto());
      graph_defs.AddGraphOutputs({output_name});
    }

    graph_defs.AddNodeDefs({NodeDef("AdamOptimizer",
                                    input_args,
                                    output_args,
                                    attr,
                                    "AdamOptimizer_" + weight_args[0]->Name())});
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
