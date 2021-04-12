// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "dnnl.hpp"
#include "core/providers/dnnl/subgraph/subgraph.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlKernel {
 public:
  DnnlKernel(const DnnlNode& node,
             DNNLExecutionProvider* provider) {
    name_ = node.name;
    mklnode_ptr_ = std::make_shared<DnnlNode>(node);
    provider_ = provider;
    alloc_ = provider_->GetAllocator(0, OrtMemTypeDefault);
  }
  virtual ~DnnlKernel(){};

  virtual void CreatePrimitives(const OrtCustomOpApi* api,
                                OrtKernelContext* context,
                                const std::unordered_map<dnnl::engine::kind, dnnl::engine>& dnnl_engine,
                                std::vector<dnnl::primitive>& net,
                                std::vector<std::unordered_map<int, dnnl::memory>>& net_args) = 0;

  virtual void ReorderWeights(const OrtCustomOpApi* api, OrtKernelContext* context, const dnnl::engine& cpu_engine) {
    ORT_UNUSED_PARAMETER(api);
    ORT_UNUSED_PARAMETER(context);
    ORT_UNUSED_PARAMETER(cpu_engine);
  }
  virtual void SetProvider(DNNLExecutionProvider* provider) {
    provider_ = provider;
  }
  DNNLExecutionProvider* GetProvider() {
    return provider_;
  }

  virtual Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) = 0;

 protected:
  virtual void ReadAttributes(const NodeAttributes& attributes,
                              const std::string attributes_prefix = "") {
    ORT_UNUSED_PARAMETER(attributes);
    ORT_UNUSED_PARAMETER(attributes_prefix);
  }

  Status GetIntsAttr(const ONNX_NAMESPACE::AttributeProto& proto, std::vector<int64_t>& values) {
    ORT_RETURN_IF_NOT(proto.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS,
                      "proto.type() != AttributeProto_AttributeType_INTS");
    values.reserve(proto.ints_size());
    for (int i = 0; i < proto.ints_size(); i++) {
      values.push_back(proto.ints(i));
    }
    return Status::OK();
  }

  Status GetIntAttr(const ONNX_NAMESPACE::AttributeProto& proto, int64_t& value) {
    ORT_RETURN_IF_NOT(proto.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT,
                      "proto.type() != AttributeProto_AttributeType_INT");
    value = proto.i();
    return Status::OK();
  }

  Status GetFloatAttr(const ONNX_NAMESPACE::AttributeProto& proto, float& value) {
    ORT_RETURN_IF_NOT(proto.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT,
                      "proto.type() != AttributeProto_AttributeType_FLOAT");
    value = proto.f();
    return Status::OK();
  }
  Status GetStringAttr(const ONNX_NAMESPACE::AttributeProto& proto, std::string& value) {
    ORT_RETURN_IF_NOT(proto.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING,
                      "proto.type() != AttributeProto_AttributeType_STRING");
    value = proto.s();
    return Status::OK();
  }

  void InitDstReorderOutput(dnnl::engine& cpu_engine,
                            dnnl::memory::data_type& data_type,
                            std::vector<dnnl::primitive>& net,
                            std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
                            bool gpu_available = false);

  dnnl::memory::format_tag GetSourceFormat(int dim_size);

 public:
  std::string name_;
  std::vector<std::shared_ptr<DnnlKernel>> parents_;
  bool fuse_relu_ = false;
  bool fuse_sum_ = false;
  std::shared_ptr<dnnl::memory> primitive_dst_mem_;
  std::unique_ptr<dnnl::memory::desc> primitive_dst_md_;
  TensorShape primitive_dst_shape_;
  dnnl::memory::desc primitive_dst_desc_;
  // ONNX Runtime format
  dnnl::memory::format_tag ort_source_format_;
  dnnl::memory::desc ort_source_desc_;

  // input format.
  // It can be ORT format (nchw) or blocked memory format from parent node
  // dnnl::memory::format_tag source_format_ = dnnl::memory::format_tag::any;
  dnnl::memory::desc source_desc_ = dnnl::memory::desc();
  Status primitive_created_status_;

 protected:
  // Pointer to MklNode of subgraph IR
  std::shared_ptr<DnnlNode> mklnode_ptr_;
  // input format expected by primitive object
  dnnl::memory::desc primitive_src_desc_;

  // memory used for reorders
  std::unique_ptr<dnnl::memory> reorder_dst_mem_to_;
  AllocatorPtr alloc_;
  DNNLExecutionProvider* provider_;
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
