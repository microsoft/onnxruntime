// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/ort_value.h"
#include "core/graph/abi_graph_types.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

/// <summary>
/// Concrete implementation of OrtValueInfo used in the ModelEditorApi.
/// </summary>
struct ModelEditorValueInfo : public OrtValueInfo {
  ModelEditorValueInfo() : OrtValueInfo(OrtGraphIrApi::kModelEditorApi) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtValueInfo and ModelEditorValueInfo.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtValueInfo, ModelEditorValueInfo, OrtGraphIrApi::kModelEditorApi)

  const std::string& GetName() const override { return name; }

  const OrtTypeInfo* GetTypeInfo() const override { return type_info.get(); }

  Status GetProducerInfo(ProducerInfo& /*producer_info*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the producer for OrtValueInfo");
  }

  Status GetConsumerInfos(std::vector<OrtValueInfo::ConsumerInfo>& /*consumer_infos*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the consumers for a OrtValueInfo");
  }

  Status GetNumConsumerInfos(size_t& /*num_consumers*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the number of consumers for a OrtValueInfo");
  }

  Status GetInitializerValue(const OrtValue*& /*value*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the initializer value for a OrtValueInfo");
  }

  Status IsRequiredGraphInput(bool& /*is_required_graph_input*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support querying if a graph input is required for OrtValueInfo");
  }

  Status IsOptionalGraphInput(bool& /*is_optional_graph_input*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support querying if OrtValueInfo is an optional graph input.");
  }

  Status IsGraphOutput(bool& /*is_graph_output*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support querying if a OrtValueInfo is a graph output.");
  }

  Status IsConstantInitializer(bool& /*is_const_initializer*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support querying if a OrtValueInfo is a constant initializer.");
  }

  Status IsFromOuterScope(bool& /*is_outer_scope*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support querying if a OrtValueInfo is defined in an outer scope.");
  }

  std::string name;
  std::unique_ptr<OrtTypeInfo> type_info;
};

/// <summary>
/// Concrete implementation of OrtNode used in the ModelEditorApi.
/// </summary>
struct ModelEditorNode : public OrtNode {
  ModelEditorNode() : OrtNode(OrtGraphIrApi::kModelEditorApi) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtNode and ModelEditorNode.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtNode, ModelEditorNode, OrtGraphIrApi::kModelEditorApi)

  size_t GetId() const override { return id; }

  const std::string& GetName() const override { return node_name; }

  const std::string& GetOpType() const override { return operator_name; }

  const std::string& GetDomain() const override { return domain_name; }

  Status GetSinceVersion(int& /*since_version*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting an OrtNode's opset version");
  }

  size_t GetNumInputs() const override { return input_names.size(); }

  Status GetInputs(gsl::span<const OrtValueInfo*> /*inputs*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting input OrtValueInfos for OrtNode");
  }

  size_t GetNumOutputs() const override { return output_names.size(); }

  Status GetOutputs(gsl::span<const OrtValueInfo*> /*outputs*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting output OrtValueInfos for OrtNode");
  }

  Status GetNumImplicitInputs(size_t& /*num_implicit_inputs*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the implicit inputs for OrtNode");
  }

  Status GetImplicitInputs(gsl::span<const OrtValueInfo*> /*implicit_inputs*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the implicit inputs for OrtNode");
  }

  size_t GetNumAttributes() const override { return attributes.size(); }

  Status GetAttributes(gsl::span<const OrtOpAttr*> /*attrs*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting attribute OrtOpAttr for OrtNode");
  }

  Status GetNumSubgraphs(size_t& /*num_subgraphs*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the subgraphs for OrtNode");
  }

  Status GetSubgraphs(gsl::span<const OrtGraph*> /*subgraphs*/,
                      const char** /*opt_attribute_names*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the subgraphs for OrtNode");
  }

  Status GetGraph(const OrtGraph*& /*parent_graph*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the parent graph for OrtNode");
  }

  size_t id = 0;
  std::string operator_name;
  std::string domain_name;
  std::string node_name;

  // OrtOpAttr is 1:1 with ONNX_NAMESPACE::AttributeProto currently.
  // https://github.com/microsoft/onnxruntime/blob/bd5a759d0cdbed6e7f611c990d4eb5457a9ecf60/onnxruntime/core/session/standalone_op_invoker.cc#L318
  onnxruntime::InlinedVector<ONNX_NAMESPACE::AttributeProto> attributes;
  onnxruntime::InlinedVector<std::string> input_names;
  onnxruntime::InlinedVector<std::string> output_names;

  // FUTURE if we need control flow nodes
  // std::unordered_map<std::string, OrtGraph> subgraphs;
};

/// <summary>
/// Concrete implementation of OrtGraph used in the ModelEditorApi.
/// </summary>
struct ModelEditorGraph : public OrtGraph {
  ModelEditorGraph() : OrtGraph(OrtGraphIrApi::kModelEditorApi) {}

  // Defines ToExternal() and ToInternal() functions to convert between OrtGraph and ModelEditorGraph.
  DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(OrtGraph, ModelEditorGraph, OrtGraphIrApi::kModelEditorApi)

  const std::string& GetName() const override { return name; }

  const ORTCHAR_T* GetModelPath() const override { return ORT_TSTR(""); }

  int64_t GetOnnxIRVersion() const override {
    return ONNX_NAMESPACE::Version::IR_VERSION;
  }

  Status GetNumOperatorSets(size_t& /*num_operator_sets*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the graph's operator sets.");
  }

  Status GetOperatorSets(gsl::span<const char*> /*domains*/,
                         gsl::span<int64_t> /*opset_versions*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the graph's operator sets.");
  }

  size_t GetNumInputs() const override { return inputs.size(); }

  Status GetInputs(gsl::span<const OrtValueInfo*> /*result*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the graph inputs.");
  }

  size_t GetNumOutputs() const override { return outputs.size(); }

  Status GetOutputs(gsl::span<const OrtValueInfo*> /*result*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the graph outputs.");
  }

  size_t GetNumInitializers() const override { return initializers.size() + external_initializers.size(); }

  Status GetInitializers(gsl::span<const OrtValueInfo*> /*result*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the graph initializers.");
  }

  size_t GetNumNodes() const override { return nodes.size(); }

  Status GetNodes(gsl::span<const OrtNode*> /*result*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the graph nodes.");
  }

  Status GetParentNode(const OrtNode*& /*parent_node*/) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "OrtModelEditorApi does not support getting the parent node for OrtGraph");
  }

  onnxruntime::InlinedVector<std::unique_ptr<onnxruntime::ModelEditorValueInfo>> inputs;
  onnxruntime::InlinedVector<std::unique_ptr<onnxruntime::ModelEditorValueInfo>> outputs;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> initializers;
  std::unordered_map<std::string, std::unique_ptr<OrtValue>> external_initializers;
  std::vector<std::unique_ptr<onnxruntime::ModelEditorNode>> nodes;
  std::string name = "ModelEditorGraph";
};

}  // namespace onnxruntime

struct OrtModel {
  std::unique_ptr<OrtGraph> graph;
  std::unordered_map<std::string, int> domain_to_version;
};
