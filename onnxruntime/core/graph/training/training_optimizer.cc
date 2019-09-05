// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/graph/training/training_optimizer.h"
#include "core/graph/training/gradient_builder_base.h"
#include "core/graph/training/graph_augmenter.h"

namespace onnxruntime {
namespace training {

#ifdef USE_HOROVOD
static const std::string global_barrier_name = "horovod/barrier";
static const std::string global_barrier_ready = "horovod/barrier/ready";

static NodeDef BuildGlobalBarrierNode(const std::vector<std::string>& ready_names, GraphAugmenter::GraphDefs& graph_defs) {
  std::string barrier_input_name = global_barrier_name + "/input";
  std::string barrier_output_name = global_barrier_name + "/output";

  // Global barrier no-op input.
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.add_dims(0);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_proto.set_name(barrier_input_name);
  graph_defs.AddInitializers({tensor_proto});

  std::vector<ArgDef> barrier_inputs{barrier_input_name};
  std::transform(ready_names.begin(), ready_names.end(), std::back_inserter(barrier_inputs), [](const std::string& name) { return ArgDef(name); });
  std::vector<ArgDef> barrier_outputs{barrier_output_name, global_barrier_ready};

  return NodeDef("HorovodBarrier", barrier_inputs, barrier_outputs, NodeAttributes(), global_barrier_name);
}

static NodeDef& GetGlobalBarrierNode(GraphAugmenter::GraphDefs& graph_defs) {
  // Find the global barrier node.
  auto& nodes = graph_defs.NodeDefs();
  auto barrier_iter = std::find_if(nodes.begin(), nodes.end(), [&](const NodeDef& def) { return def.name == global_barrier_name; });
  if (barrier_iter != nodes.end())
    return *barrier_iter;

  // Create the global barrier.
  graph_defs.AddNodeDefs({BuildGlobalBarrierNode({}, graph_defs)});
  return *std::find_if(nodes.begin(), nodes.end(), [&](const NodeDef& def) { return def.name == global_barrier_name; });
}

static std::string BuildAllReduceNode(const std::string& gradient, GraphAugmenter::GraphDefs& graph_defs) {
  ArgDef reduce_output(gradient + "_AllReduce_Out");
  ArgDef reduce_ready(gradient + "_AllReduce_Ready");
  ArgDef local_barrier_output(gradient + "_Barrier_Out");
  ArgDef local_barrier_ready(gradient + "_Barrier_Ready");

  // Add horovod all reduce node.
  graph_defs.AddNodeDefs({NodeDef("HorovodAllReduce", {gradient}, {reduce_output, reduce_ready}, NodeAttributes(), gradient)});

  // Add ready check to global barrier.
  NodeDef& global_barrier_node = GetGlobalBarrierNode(graph_defs);
  global_barrier_node.input_args.push_back(reduce_ready);

  // Add local barrier node.
  graph_defs.AddNodeDefs({NodeDef("HorovodBarrier", {reduce_output, global_barrier_ready}, {local_barrier_output, local_barrier_ready}, NodeAttributes(), gradient + "_Barrier")});

  return local_barrier_output.name;
}
#else
static std::string BuildAllReduceNode(const std::string& /*gradient*/, GraphAugmenter::GraphDefs& /*graph_defs*/) {
  ORT_NOT_IMPLEMENTED("Distributed training is not supported, as Horovod is not enabled in this build.");
}
#endif

Status SGDOptimizerBuilder::Build(const NodeArg* weight_arg,
                                  const OptimizerInfo& opt_info,
                                  GraphAugmenter::GraphDefs& graph_defs) const {
  const std::string weight_name = weight_arg->Name();
  const TypeProto* weight_type_proto = weight_arg->TypeAsProto();
  const std::string gradient_name = GradientBuilderBase::GradientName(weight_name);

  // Initialized tensor for Learning Rate
  TensorProto lr_tensor_proto = CreateTensorProto<float>(learning_rate_name_, opt_info.learning_rate);
  graph_defs.AddInitializers({lr_tensor_proto});

  std::string accumulated_gradient_name = gradient_name;
  if (opt_info.world_size > 1) {
    accumulated_gradient_name = BuildAllReduceNode(gradient_name, graph_defs);
  }

  std::vector<ArgDef> input_args(num_inputs_);
  input_args[0] = ArgDef(learning_rate_name_);
  input_args[1] = ArgDef(weight_name, weight_type_proto);
  input_args[2] = ArgDef(accumulated_gradient_name, weight_type_proto);

  std::vector<ArgDef> output_args(num_outputs_);
  output_args[0] = ArgDef(weight_name + "_SGD_out", weight_type_proto);  // output 0 new weights

  graph_defs.AddNodeDefs({NodeDef(OpType(),
                                  input_args,
                                  output_args,
                                  NodeAttributes(),
                                  OptimizerNodeName(weight_name))});
  return Status::OK();
}

Status AdamOptimizerBuilder::Build(const NodeArg* weight_arg,
                                   const OptimizerInfo& opt_info,
                                   GraphAugmenter::GraphDefs& graph_defs) const {
  const std::string weight_name = weight_arg->Name();
  const std::string gradient_name = GradientBuilderBase::GradientName(weight_name);

  const TypeProto* weight_type_proto = weight_arg->TypeAsProto();

  // Initialized tensor for Learning Rate
  TensorProto lr_tensor_proto = CreateTensorProto<float>(learning_rate_name_, opt_info.learning_rate);

  // The type proto initializer for Update Count
  std::string update_count_string = "Update_Count_" + weight_name;  // per weight optimizer requires a per weight update count
  TensorProto uc_tensor_proto = CreateTensorProto<int64_t>(update_count_string, 1);

  // Add lr and uc tensorproto as initializers
  graph_defs.AddInitializers({lr_tensor_proto, uc_tensor_proto});

  std::string accumulated_gradient_name = gradient_name;
  if (opt_info.world_size > 1) {
    accumulated_gradient_name = BuildAllReduceNode(gradient_name, graph_defs);
  }

  int num_inputs = num_inputs_;
  int num_outputs = num_outputs_;
  // When mixed precision is enabled by using FP16 initializer, optimizer consumes fp32 weight tensor and its fp16 copy.
  // Thus, each optimizer will get one extra input and one extra output.
  if (opt_info.fp16_weight_arg != nullptr) {
    num_inputs += 1;
    num_outputs += 1;
  }
  std::vector<ArgDef> input_args(num_inputs);
  input_args[0] = ArgDef(learning_rate_name_);
  input_args[1] = ArgDef(update_count_string);
  input_args[2] = ArgDef(weight_name, weight_type_proto);
  input_args[3] = ArgDef(accumulated_gradient_name, weight_type_proto);

  std::vector<ArgDef> output_args(num_outputs);
  output_args[0] = ArgDef(weight_name + "_Adam_out", weight_type_proto);

  // The tensor proto for first and second moments of grad
  int input_idx = 4;
  int output_idx = 1;
  std::vector<std::string> moments_prefixes({"Moment_1_", "Moment_2_"});
  for (auto moments_prefix : moments_prefixes) {
    std::string gradient_moment_name = moments_prefix + gradient_name;
    std::vector<int64_t> dims;
    for (auto dim : weight_arg->Shape()->dim()) {
      dims.push_back(dim.dim_value());
    }

    TensorProto moment_tensor_proto;
    TypeProto* moment_type_proto = graph_defs.CopyTypeProto(weight_arg);
    if (opt_info.use_fp16_moments) {
      moment_tensor_proto = CreateTensorProto<MLFloat16>(gradient_moment_name, MLFloat16(math::floatToHalf(0.f)), dims);
      moment_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16);
    } else {
      moment_tensor_proto = CreateTensorProto<float>(gradient_moment_name, 0.f, dims);
    }

    graph_defs.AddInitializers({moment_tensor_proto});
    input_args[input_idx++] = ArgDef(gradient_moment_name, moment_type_proto);
    output_args[output_idx++] = ArgDef(gradient_moment_name + "_Out", moment_type_proto);
  }

  TypeProto* type_proto = graph_defs.CreateTypeProto({}, ONNX_NAMESPACE::TensorProto_DataType_INT64);
  output_args[3] = ArgDef(gradient_name + "_Step_Out", type_proto);

  if (opt_info.fp16_weight_arg != nullptr) {
    input_args[input_idx++] = ArgDef(opt_info.fp16_weight_arg->Name(), opt_info.fp16_weight_arg->TypeAsProto());

    std::string output_name = opt_info.fp16_weight_arg->Name() + "_Adam_out";
    output_args[4] = ArgDef(output_name, opt_info.fp16_weight_arg->TypeAsProto());
  }

  graph_defs.AddNodeDefs({NodeDef(OpType(),
                                  input_args,
                                  output_args,
                                  BuildAttributeProto(opt_info),
                                  OptimizerNodeName(weight_name))});
  return Status::OK();
}

Status LambOptimizerBuilder::Build(const NodeArg* weight_arg,
                                   const OptimizerInfo& opt_info,
                                   GraphAugmenter::GraphDefs& graph_defs) const {
  const std::string weight_name = weight_arg->Name();
  const std::string gradient_name = GradientBuilderBase::GradientName(weight_name);

  const TypeProto* weight_type_proto = weight_arg->TypeAsProto();

  // Initialized tensor for Learning Rate
  TensorProto lr_tensor_proto = CreateTensorProto<float>(learning_rate_name_, opt_info.learning_rate);
  graph_defs.AddInitializers({lr_tensor_proto});

  std::string accumulated_gradient_name = gradient_name;
  if (opt_info.world_size > 1) {
    accumulated_gradient_name = BuildAllReduceNode(gradient_name, graph_defs);
  }

  std::vector<ArgDef> input_args(num_inputs_);
  input_args[0] = ArgDef(learning_rate_name_);
  input_args[1] = ArgDef(weight_name, weight_type_proto);
  input_args[2] = ArgDef(accumulated_gradient_name, weight_type_proto);

  std::vector<ArgDef> output_args(num_outputs_);
  output_args[0] = ArgDef(weight_name + "_Lamb_out", weight_type_proto);

  // The tensor proto for first and second moments of grad
  int input_idx = 3;
  int output_idx = 1;
  std::vector<std::string> moments_prefixes({"Moment_1_", "Moment_2_"});
  for (auto moment_prefix : moments_prefixes) {
    std::string gradient_moment_name = moment_prefix + gradient_name;
    std::vector<int64_t> dims;
    for (auto dim : weight_arg->Shape()->dim()) {
      dims.push_back(dim.dim_value());
    }

    TensorProto moment_tensor_proto;
    TypeProto* moment_type_proto = graph_defs.CopyTypeProto(weight_arg);
    if (opt_info.use_fp16_moments) {
      moment_tensor_proto = CreateTensorProto<MLFloat16>(gradient_moment_name, MLFloat16(math::floatToHalf(0.f)), dims);
      moment_type_proto->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16);
    } else {
      moment_tensor_proto = CreateTensorProto<float>(gradient_moment_name, 0.f, dims);
    }

    graph_defs.AddInitializers({moment_tensor_proto});
    input_args[input_idx++] = ArgDef(gradient_moment_name, moment_type_proto);
    output_args[output_idx++] = ArgDef(gradient_moment_name + "_Out", moment_type_proto);
  }

  graph_defs.AddNodeDefs({NodeDef(OpType(),
                                  input_args,
                                  output_args,
                                  BuildAttributeProto(opt_info),
                                  OptimizerNodeName(weight_name))});
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
