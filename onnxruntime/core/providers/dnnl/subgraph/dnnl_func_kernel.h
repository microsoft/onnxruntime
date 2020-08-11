// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/session/onnxruntime_c_api.h"
#include "dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {

namespace {
struct SubgraphParams {
  std::unique_ptr<Provider_NodeAttributes> attributes{Provider_NodeAttributes::Create()};
  DNNLExecutionProvider* provider;
  std::shared_ptr<Subgraph> subgraph;
  std::string subgraph_id;
  std::string subgraph_key;

  SubgraphParams() {}
};
}  // namespace

template <typename T>
class DnnlFuncKernel {
 public:
  explicit DnnlFuncKernel(const ComputeContext* context,
                          const Provider_NodeAttributes& attributes,
                          DNNLExecutionProvider* provider) {
    ORT_UNUSED_PARAMETER(context);

    params_.provider = provider;
    *params_.attributes = attributes;

    auto sub_it = attributes.find("subgraph_id");
    if (sub_it->second().type() == ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
      params_.subgraph_id = sub_it->second().s();
      params_.subgraph = provider->GetDnnlSubgraph(params_.subgraph_id);

      std::ostringstream key_os;
      key_os << params_.subgraph->graph_name << "_" << params_.subgraph_id << "-";
      key_os << params_.subgraph->dnnl_nodes.back().ToString() << "-";
      key_os << params_.subgraph->dnnl_nodes.back().output_name;

      if (params_.subgraph->dnnl_nodes[0].name == "Conv") {
        std::ostringstream os;
        os << "Conv-" << params_.subgraph->dnnl_nodes[0].node_index << "-";
        key_os << GetConvAttributeKey(attributes, os.str());
      }

      if (params_.subgraph->dnnl_nodes[0].name == "LRN") {
        std::ostringstream os;
        os << "LRN-" << params_.subgraph->dnnl_nodes[0].node_index << "-";
        key_os << GetLrnAttributeKey(attributes, os.str());
      }

      if (params_.subgraph->dnnl_nodes[0].name.find("Pool") != std::string::npos) {
        std::ostringstream os;
        os << params_.subgraph->dnnl_nodes[0].name << "-" << params_.subgraph->dnnl_nodes[0].node_index << "-";
        key_os << GetPoolAttributesKey(attributes, os.str());
      }

      params_.subgraph_key = key_os.str();
    }
  }

  std::string GetPoolAttributesKey(const Provider_NodeAttributes& attributes,
                                   const std::string attributes_prefix = "") {
    std::string key;

    auto attr = attributes.find(attributes_prefix + "kernel_shape");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "auto_pad");
    if (attr != attributes.end()) {
      key.append(attr->second().s());
    }

    attr = attributes.find(attributes_prefix + "pads");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "strides");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "count_include_pad");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      key.append(std::to_string(proto.i()));
      key.append(1, '_');
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "ceil_mode");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      key.append(std::to_string(proto.i()));
      key.append(1, '_');
      key.append(1, '#');
    }
    return key;
  }

  std::string GetConvAttributeKey(const Provider_NodeAttributes& attributes,
                                  const std::string attributes_prefix = "") {
    std::string key;

    auto attr = attributes.find(attributes_prefix + "dilations");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "auto_pad");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      key.append(proto.s());
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "pads");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "strides");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "kernel_shape");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "group");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      key.append(std::to_string(proto.i()));
      key.append(1, '#');
    }

    return key;
  }

  std::string GetLrnAttributeKey(const Provider_NodeAttributes& attributes,
                                 const std::string attributes_prefix = "") {
    std::string key;

    auto attr = attributes.find(attributes_prefix + "alpha");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      key.append(std::to_string(proto.f()));
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "beta");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      key.append(std::to_string(proto.f()));
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "bias");
    if (attr != attributes.end()) {
      key.append(1, '#');
      auto& proto = attr->second();
      key.append(std::to_string(proto.f()));
      key.append(1, '#');
    }

    return key;
  }

  Status Compute(const OrtCustomOpApi* api, OrtKernelContext* context) const;

 private:
  SubgraphParams params_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
