// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/training/generic_registry.h"
#include "core/graph/training/graph_augmenter.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
namespace training {
namespace in_graph_optimizer {

struct OptimizerInfo {
  std::string name_;
  float learning_rate;
  int world_rank;
  int world_size;
  std::unordered_map<std::string, float> attributes_;
};

// Utils for Constant Node Creation - Currently Used only in in_graph_training_optimizer
// TODO: Move to an appropriate place if need arises
// TODO: Add more types as necessary
template <typename T>
inline void SetTypedDataToTensor(T val, TensorProto& tensor, int64_t count);

template <>
inline void SetTypedDataToTensor<float>(float val, TensorProto& tensor, int64_t count) {
  for (int64_t i = 0; i < count; i++) {
    tensor.add_float_data(val);
  }
}

template <>
inline void SetTypedDataToTensor<int64_t>(int64_t val, TensorProto& tensor, int64_t count) {
  for (int64_t i = 0; i < count; i++) {
    tensor.add_int64_data(val);
  }
}

template <class T>
TensorProto CreateTensorProto(
    std::string name,
    T val,
    std::vector<int64_t> dims = {1}) {
  TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(data_types_internal::ToTensorDataType<T>());
  std::for_each(dims.begin(), dims.end(), [&](auto dim) { tensor_proto.add_dims(dim); });
  SetTypedDataToTensor<T>(val, tensor_proto, std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{}));
  return tensor_proto;
}

class OptimizerBuilder {
 public:
  OptimizerBuilder(const std::string name) : name_(name) {}

  virtual ~OptimizerBuilder() {}

  virtual common::Status Build(const std::vector<const NodeArg*> weight_args,
                               const OptimizerInfo& opt_info,
                               GraphAugmenter::GraphDefs& graph_defs) const = 0;

  const std::string& Name() const { return name_; }

 protected:
  common::Status BuildAllReduceNode(
      const std::vector<std::string>& gradients,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ArgDef>& agg_grads) const {
    // inputs : grads
    std::vector<ArgDef> input_grads;
    for (const auto& grad : gradients) {
      input_grads.emplace_back(grad, nullptr);
    }

    // outputs: aggregated grads
    agg_grads.clear();
    for (const auto& grad : gradients) {
      std::string output_name = grad + "_AllReduce_Out";
      agg_grads.emplace_back(output_name, nullptr);
    }

    // The following code works under the assumption allreduce is called per gradient.
    // TODO: if the assumption changes, the parameter to AddNodeDefs(gradients[0]) needs to be fixed
    ORT_ENFORCE(gradients.size() == 1);
    graph_defs.AddNodeDefs({NodeDef("HorovodAllReduceOp", input_grads, agg_grads, NodeAttributes(), gradients[0])});
    return Status::OK();
  }

  const std::string learning_rate_string_ = "Learning_Rate";

 private:
  std::string name_;
};

class OptimizerBuilderRegistry : public GenericRegistry<OptimizerBuilder> {
 public:
  // Register optimizer builders.
  void RegisterBuilders();

  static OptimizerBuilderRegistry& GetInstance() {
    static OptimizerBuilderRegistry instance;
    return instance;
  }

 private:
  OptimizerBuilderRegistry() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerBuilderRegistry);
};

}  // namespace in_graph_optimizer
}  // namespace training
}  // namespace onnxruntime
