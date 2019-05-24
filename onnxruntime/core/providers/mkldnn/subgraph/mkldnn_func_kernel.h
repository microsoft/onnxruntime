// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/func_api.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
struct SubgraphParams {
  NodeAttributes attributes;
  MKLDNNExecutionProvider* provider;
  std::shared_ptr<Subgraph> subgraph;
  std::string subgraph_id;
  std::string subgraph_key;

  SubgraphParams() {}
};
}  // namespace

template <typename T>
class MkldnnFuncKernel {
 public:
  explicit MkldnnFuncKernel(const ComputeContext* context,
                          const NodeAttributes& attributes,
                          MKLDNNExecutionProvider* provider) {
    params_.provider = provider;
    params_.attributes = attributes;

    auto sub_it = attributes.find("subgraph_id");
    if (sub_it->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
      params_.subgraph_id = sub_it->second.s();
      params_.subgraph = provider->GetMklDnnSubgraph(params_.subgraph_id);
      std::ostringstream key_os;
      key_os << params_.subgraph_id << "-" << params_.subgraph->mklnodes.back().name << "-" << params_.subgraph->mklnodes.back().output_name;
      params_.subgraph_key = key_os.str();
    }
  }

  Status Compute(const OrtCustomOpApi* api, OrtKernelContext* context) const;

 private:
  SubgraphParams params_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime