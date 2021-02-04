// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/optional.h"
#include "core/framework/data_types.h"
#include "orttraining/core/graph/generic_registry.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "onnx/defs/attr_proto_util.h"
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace training {

const std::vector<std::string> MOMENTS_PREFIXES({"Moment_1", "Moment_2"});
const std::string LAMB_STEP_TENSOR_NAME = "Step";
const std::string ADAM_UC_PREFIX = "Update_Count";

template <class T>
ONNX_NAMESPACE::TensorProto CreateTensorProto(
    const std::string& name,
    T val,
    const std::vector<int64_t>& dims = {1}) {
  size_t count = static_cast<size_t>(std::accumulate(dims.begin(), dims.end(), int64_t(1), std::multiplies<int64_t>{}));
  std::vector<T> values(count, val);
  ONNX_NAMESPACE::TensorProto tensor_proto = ONNX_NAMESPACE::ToTensor<T>(values);
  tensor_proto.set_name(name);
  std::for_each(dims.begin(), dims.end(), [&](auto dim) { tensor_proto.add_dims(dim); });
  return tensor_proto;
}

template <class T>
ONNX_NAMESPACE::TensorProto CreateTensorProto(
    const std::string& name,
    const std::vector<T>& values,
    const std::vector<int64_t>& dims = {1}) {
  size_t count = static_cast<size_t>(std::accumulate(dims.begin(), dims.end(), int64_t(1), std::multiplies<int64_t>{}));
  ORT_ENFORCE(values.size() == count);
  ONNX_NAMESPACE::TensorProto tensor_proto = ONNX_NAMESPACE::ToTensor<T>(values);
  tensor_proto.set_name(name);
  std::for_each(dims.begin(), dims.end(), [&](auto dim) { tensor_proto.add_dims(dim); });
  return tensor_proto;
}

Status IsMatchingTypeAndShape(
    const onnxruntime::Tensor& tensor,
    const int32_t element_type,
    const std::vector<int64_t>& expected_shape);

/**
   * The configuration for optimizer builder.
   */
struct OptimizerBuilderConfig {
  //The ArgDefs of the weights to optimize.
  std::vector<ArgDef> weight_argdefs;

  // The ArgDefs of the gradient of the weight to
  // optimize.
  std::vector<ArgDef> gradient_argdefs;

  // (Optional) The ArgDef of gradient norm.
  optional<ArgDef> gradient_norm_argdef;

  // (Optional) The ArgDef indicates whether
  //  the passed-in gradient norm is finite.
  optional<ArgDef> gradient_norm_finite_argdef;

  // The per weight optimizer configuration.
  std::vector<OptimizerNodeConfig> opt_configs;

  // (Optional) The flag to force gradient clipping. If planning
  // to use the default behavior of each sub-class, should not be set.
  optional<bool> enable_grad_clipping;

  // The initial state for optimizer params
  // shared by all weights.
  NameMLValMap shared_optimizer_states{};
};

class OptimizerBuilder {
 public:
  OptimizerBuilder(const OpDef& op_def, const std::vector<std::string>& attribute_names = {})
      : op_def_(op_def),
        attr_names_(attribute_names) {}

  virtual ~OptimizerBuilder() {}

  /**
   * Adds the optimizer node to the graph.
   * This component may be placed into an If node subgraph to enable
   * conditional execution.
   *
   * @param config The input config for optimizer builder
   * @param[out] graph_defs The GraphDefs corresponding to the graph (possibly
   *             a subgraph) that the component is to be added to.
   * @param[out] new_initializers Any initializers that should be
   *             placed in the parent graph, if there is one.
   *             Other initializers are treated as local to the current
   *             (sub)graph. 
   * @param[out] weight_to_opt_mapping Mapping between weight to 
   *             their new optimizer states in new_initializers.
   * @param[out] output_weight_argdefs The output weight ArgDef. All optimizers
                 should have this output.
   * @param[out] output_gradient_argdefs The output gradient ArgDef. All optimizers
                 should have this output.
   *
   * @return The status of the optimizer node addition.
   */
  virtual Status Build(
      const OptimizerBuilderConfig& config,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<TensorProto>& new_initializers,
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
      std::vector<ArgDef>& output_weight_argdefs,
      std::vector<ArgDef>& output_gradient_argdefs) const = 0;

  const OpDef& OpDefinition() const { return op_def_; }

 protected:
  const std::string OptimizerNodeName(const std::string& weight_name) const {
    return op_def_.type + "_" + weight_name;
  }

  std::vector<ONNX_NAMESPACE::AttributeProto> BuildAttributeProto(const OptimizerNodeConfig& opt_config) const {
    std::vector<ONNX_NAMESPACE::AttributeProto> attribute_protos;
    for (auto attr_name : attr_names_) {
      // Search dictionary of float attributes.
      if (opt_config.attributes.count(attr_name)) {
        attribute_protos.push_back(ONNX_NAMESPACE::MakeAttribute(attr_name, opt_config.attributes.at(attr_name)));
      }
      // Search dictionary of int attributes.
      if (opt_config.int_attributes.count(attr_name)) {
        attribute_protos.push_back(ONNX_NAMESPACE::MakeAttribute(attr_name, opt_config.int_attributes.at(attr_name)));
      }
    }
    return attribute_protos;
  }

  static const ONNX_NAMESPACE::TypeProto* CreateLearningRateTypeProto(GraphAugmenter::GraphDefs& graph_defs) {
    return graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  }

 private:
  OpDef op_def_;
  std::vector<std::string> attr_names_;
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
