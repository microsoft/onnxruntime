// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/basic_types.h"

#define DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(external_type, internal_type, internal_api) \
  external_type* ToExternal() { return static_cast<external_type*>(this); }                        \
  const external_type* ToExternal() const { return static_cast<const external_type*>(this); }      \
  static internal_type* ToInternal(external_type* e) {                                             \
    return e->graph_ir_api == (internal_api) ? static_cast<internal_type*>(e) : nullptr;           \
  }                                                                                                \
  static const internal_type* ToInternal(const external_type* e) {                                 \
    return e->graph_ir_api == (internal_api) ? static_cast<const internal_type*>(e) : nullptr;     \
  }

// The public ORT graph IR types (e.g., OrtGraph, OrtNode, etc.) have different implementations for the
// ModelEditor API and EP API. This enum allows a user of the base class (e.g., OrtGraph) to determine
// the API for which the derived class was created.
enum class OrtGraphIrApi {
  kInvalid = 0,
  kModelEditorApi,
  kEpApi,
};

/// <summary>
/// Public type that represents an ONNX value info.
/// </summary>
struct OrtValueInfo {
  explicit OrtValueInfo(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtValueInfo() = default;

  virtual const std::string& Name() const = 0;
  virtual const OrtTypeInfo* TypeInfo() const = 0;

  struct ProducerInfo {
    ProducerInfo() = default;
    ProducerInfo(const OrtNode* node, size_t output_index) : node(node), output_index(output_index) {}
    const OrtNode* node = nullptr;
    size_t output_index = 0;
  };
  virtual onnxruntime::Status GetProducerInfo(ProducerInfo& producer_info) const = 0;

  struct ConsumerInfo {
    ConsumerInfo() = default;
    ConsumerInfo(const OrtNode* node, int64_t input_index) : node(node), input_index(input_index) {}
    const OrtNode* node = nullptr;
    int64_t input_index = 0;  // Negative if it is an implicit input to a node that contains a subgraph (e.g., If).
  };
  virtual onnxruntime::Status GetConsumerInfos(std::vector<ConsumerInfo>& consumer_infos) const = 0;
  virtual onnxruntime::Status GetNumConsumerInfos(size_t& num_consumers) const = 0;
  virtual onnxruntime::Status GetInitializerValue(const OrtValue*& value) const = 0;

  virtual bool IsGraphInput() const = 0;
  virtual bool IsGraphOutput() const = 0;
  virtual bool IsInitializer() const = 0;
  virtual bool IsFromOuterScope() const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};

/// <summary>
/// Public type that represents an ONNX attribute. Currently, an OrtOpAttr is interchangeable with AttributeProto.
/// </summary>
struct OrtOpAttr {
  ONNX_NAMESPACE::AttributeProto attr_proto;
};

/// <summary>
/// Public type that represents an ONNX node.
/// </summary>
struct OrtNode {
  explicit OrtNode(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtNode() = default;

  virtual size_t Id() const = 0;
  virtual const std::string& Name() const = 0;
  virtual const std::string& OpType() const = 0;
  virtual const std::string& Domain() const = 0;
  virtual onnxruntime::Status GetSinceVersion(int& since_version) const = 0;
  virtual size_t NumInputs() const = 0;
  virtual size_t NumOutputs() const = 0;
  virtual onnxruntime::Status GetInputs(onnxruntime::InlinedVector<const OrtValueInfo*>& inputs) const = 0;
  virtual onnxruntime::Status GetOutputs(onnxruntime::InlinedVector<const OrtValueInfo*>& outputs) const = 0;
  virtual onnxruntime::Status GetNumImplicitInputs(size_t& num_implicit_inputs) const = 0;
  virtual onnxruntime::Status GetImplicitInputs(onnxruntime::InlinedVector<const OrtValueInfo*>& inputs) const = 0;
  virtual onnxruntime::Status GetNumAttributes(size_t& num_attrs) const = 0;
  virtual onnxruntime::Status GetAttributes(onnxruntime::InlinedVector<const OrtOpAttr*>& attrs) const = 0;
  virtual onnxruntime::Status GetNumSubgraphs(size_t& num_subgraphs) const = 0;
  virtual onnxruntime::Status GetSubgraphs(onnxruntime::InlinedVector<const OrtGraph*>& subgraphs) const = 0;
  virtual onnxruntime::Status GetParentGraph(const OrtGraph*& parent_graph) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};

/// <summary>
/// Public type that represents an ONNX graph.
/// </summary>
struct OrtGraph {
  explicit OrtGraph(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtGraph() = default;

  virtual const std::string& Name() const = 0;
  virtual int64_t OnnxIRVersion() const = 0;
  virtual size_t NumInputs() const = 0;
  virtual size_t NumOutputs() const = 0;
  virtual size_t NumInitializers() const = 0;
  virtual onnxruntime::Status GetInputs(onnxruntime::InlinedVector<const OrtValueInfo*>& inputs) const = 0;
  virtual onnxruntime::Status GetOutputs(onnxruntime::InlinedVector<const OrtValueInfo*>& outputs) const = 0;
  virtual onnxruntime::Status GetInitializers(std::vector<const OrtValueInfo*>& initializers) const = 0;
  virtual size_t NumNodes() const = 0;
  virtual std::vector<const OrtNode*> GetNodes() const = 0;
  virtual onnxruntime::Status GetParentNode(const OrtNode*& parent_node) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};
