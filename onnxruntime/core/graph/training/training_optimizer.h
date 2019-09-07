// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/training/attr_proto_util.h"
#include "core/graph/training/generic_registry.h"
#include "core/graph/training/graph_augmenter.h"
#include "core/framework/data_types.h"
#include "core/training/optimizer_config.h"

namespace onnxruntime {
namespace training {

// Utils for Constant Node Creation - Currently Used only in in_graph_training_optimizer
// TODO: Move to an appropriate place if need arises
// TODO: Add more types as necessary
template <typename T>
inline void SetTypedDataToTensor(T val, TensorProto& tensor, int64_t count);

template <>
inline void SetTypedDataToTensor<MLFloat16>(MLFloat16 val, TensorProto& tensor, int64_t count) {
  for (int64_t i = 0; i < count; i++) {
    tensor.add_int32_data(val.val);
  }
}

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

// TODO clean up if not needed - temporarily used in OptimizerGraphBuilder to set the mixed precision conditional optimizer condition to a hard-coded value
template <>
inline void SetTypedDataToTensor<bool>(bool val, TensorProto& tensor, int64_t count) {
  std::vector<char> value_bytes(count, val ? '\1' : '\0');
  tensor.set_raw_data(value_bytes.data(), count);
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
  OptimizerBuilder(const std::string& name, int num_inputs, int num_outputs,
                   const std::vector<std::string>& attribute_names = {})
      : num_inputs_(num_inputs),
        num_outputs_(num_outputs),
        name_(name),
        attr_names_(attribute_names) {}

  virtual ~OptimizerBuilder() {}

  /**
   * Adds the optimizer node to the graph.
   * This component may be placed into an If node subgraph to enable
   * conditional execution.
   *
   * @param weight_argdef The ArgDef of the weight to optimize.
   * @param gradient_argdef The ArgDef of the gradient of the weight to
            optimize.
   * @param opt_config The optimizer configuration.
   * @param[out] graph_defs The GraphDefs corresponding to the graph (possibly
   *             a subgraph) that the component is to be added to.
   * @param[out] external_inputs_including_initializers Any inputs that should
   *             come from the parent graph, if there is one.
   *             Other inputs are treated as local to the current (sub)graph.
   * @param[out] new_external_initializers Any initializers that should be
   *             placed in the parent graph, if there is one.
   *             Other initializers are treated as local to the current
   *             (sub)graph.
   * @param[out] output_weight_argdef The output weight ArgDef. All optimizers
                 should have this output.
   *
   * @return The status of the optimizer node addition.
   */
  virtual Status Build(
      const ArgDef& weight_argdef,
      const ArgDef& gradient_argdef,
      const OptimizerNodeConfig& opt_config,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ArgDef>& external_inputs_including_initializers,
      std::vector<TensorProto>& new_external_initializers,
      ArgDef& output_weight_argdef) const = 0;

  const std::string& OpType() const { return name_; }

 protected:
  const std::string learning_rate_name_ = "Learning_Rate";
  int num_inputs_;
  int num_outputs_;

  const std::string OptimizerNodeName(const std::string& weight_name) const {
    return name_ + "_" + weight_name;
  }

  std::vector<AttributeProto> BuildAttributeProto(const OptimizerNodeConfig& opt_config) const {
    std::vector<AttributeProto> attribute_protos;
    for (auto attr_name : attr_names_) {
      if (opt_config.attributes.count(attr_name)) {
        attribute_protos.push_back(MakeAttribute(attr_name, opt_config.attributes.at(attr_name)));
      }
    }
    return attribute_protos;
  }

 private:
  std::string name_;
  std::vector<std::string> attr_names_;
};

class SGDOptimizerBuilder final : public OptimizerBuilder {
 public:
  SGDOptimizerBuilder() : OptimizerBuilder("SGDOptimizer", 3, 1) {}

  virtual Status Build(
      const ArgDef& weight_argdef,
      const ArgDef& gradient_argdef,
      const OptimizerNodeConfig& opt_config,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ArgDef>& external_inputs_including_initializers,
      std::vector<TensorProto>& new_external_initializers,
      ArgDef& output_weight_argdef) const override;
};

class AdamOptimizerBuilder final : public OptimizerBuilder {
 public:
  AdamOptimizerBuilder() : OptimizerBuilder("AdamOptimizer", 6, 4,
                                            {"alpha", "beta", "lambda", "epsilon"}) {}

  virtual Status Build(
      const ArgDef& weight_argdef,
      const ArgDef& gradient_argdef,
      const OptimizerNodeConfig& opt_config,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ArgDef>& external_inputs_including_initializers,
      std::vector<TensorProto>& new_external_initializers,
      ArgDef& output_weight_argdef) const override;
};

class LambOptimizerBuilder final : public OptimizerBuilder {
 public:
  LambOptimizerBuilder() : OptimizerBuilder("LambOptimizer", 5, 3,
                                            {"alpha", "beta", "lambda", "epsilon"}) {}

  virtual Status Build(
      const ArgDef& weight_argdef,
      const ArgDef& gradient_argdef,
      const OptimizerNodeConfig& opt_config,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ArgDef>& external_inputs_including_initializers,
      std::vector<TensorProto>& new_external_initializers,
      ArgDef& output_weight_argdef) const override;
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

}  // namespace training
}  // namespace onnxruntime
