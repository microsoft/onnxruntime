// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"

#define DEFINE_ORT_GRAPH_IR_TO_EXTERNAL_INTERNAL_FUNCS(external_type, internal_type, internal_api) \
  external_type* ToExternal() { return static_cast<external_type*>(this); }                        \
  const external_type* ToExternal() const { return static_cast<const external_type*>(this); }      \
  static internal_type* ToInternal(external_type* e) {                                             \
    return e->graph_ir_api == (internal_api) ? static_cast<internal_type*>(e) : nullptr;           \
  }                                                                                                \
  static const internal_type* ToInternal(const external_type* e) {                                 \
    return e->graph_ir_api == (internal_api) ? static_cast<const internal_type*>(e) : nullptr;     \
  }

// ORT C interface types for OrtGraphApi can't be in a namespace.
// We need to define them here so onnxruntime::Model can be created from OrtModel.

enum class OrtGraphIrApi {
  kInvalid = 0,
  kModelEditorApi,
  kEpApi,
};

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

  struct UseInfo {
    UseInfo() = default;
    UseInfo(const OrtNode* node, size_t input_index) : node(node), input_index(input_index) {}
    const OrtNode* node = nullptr;
    size_t input_index = 0;
  };
  virtual onnxruntime::Status GetUses(std::vector<UseInfo>& uses) const = 0;
  virtual onnxruntime::Status GetNumUses(size_t& num_consumers) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};

struct OrtOpAttr {
  ONNX_NAMESPACE::AttributeProto attr_proto;
};

struct OrtNode {
  explicit OrtNode(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtNode() = default;

  virtual const std::string& Name() const = 0;
  virtual const std::string& OpType() const = 0;
  virtual const std::string& Domain() const = 0;
  virtual size_t GetNumInputs() const = 0;
  virtual size_t GetNumOutputs() const = 0;
  virtual onnxruntime::Status GetInputs(onnxruntime::InlinedVector<const OrtValueInfo*>& inputs) const = 0;
  virtual onnxruntime::Status GetOutputs(onnxruntime::InlinedVector<const OrtValueInfo*>& outputs) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};

struct OrtGraph {
  explicit OrtGraph(OrtGraphIrApi graph_ir_api) : graph_ir_api(graph_ir_api) {}
  virtual ~OrtGraph() = default;

  virtual const std::string& Name() const = 0;
  virtual size_t NumberOfNodes() const = 0;
  virtual std::vector<const OrtNode*> GetNodes(int order) const = 0;

  OrtGraphIrApi graph_ir_api = OrtGraphIrApi::kInvalid;
};
