// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "mkldnn.hpp"
#include "core/common/cpuid_info.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/mkldnn/subgraph/subgraph.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"

namespace onnxruntime {
namespace mkl_dnn {

class MklDnnKernel {
 public:
  MklDnnKernel(const MklDnnNode& node,
               MKLDNNExecutionProvider* provider) {
    name_ = node.name;
    mklnode_ptr_ = std::make_shared<MklDnnNode>(node);
    provider_ = provider;
    alloc_ = provider_->GetAllocator(0, OrtMemTypeDefault);
  }
  virtual ~MklDnnKernel(){};

  virtual void CreatePrimitives(const OrtCustomOpApi* api,
                                OrtKernelContext* context,
                                mkldnn::engine& cpu_engine,
                                std::vector<mkldnn::primitive>& net,
                                std::vector<std::unordered_map<int, mkldnn::memory>>& net_args) = 0;

  virtual void ReorderWeights(const OrtCustomOpApi* api, OrtKernelContext* context, mkldnn::engine& cpu_engine) {
    ORT_UNUSED_PARAMETER(api);
    ORT_UNUSED_PARAMETER(context);
    ORT_UNUSED_PARAMETER(cpu_engine);
  }
  virtual void SetProvider(MKLDNNExecutionProvider* provider) {
    provider_ = provider;
  }
  MKLDNNExecutionProvider* GetProvider() {
    return provider_;
  }

  virtual Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) = 0;

 protected:
  virtual void ReadAttributes(const NodeAttributes& attributes,
                              const std::string attributes_prefix = "") {
    ORT_UNUSED_PARAMETER(attributes);
    ORT_UNUSED_PARAMETER(attributes_prefix);
  }

  Status GetIntsAttr(ONNX_NAMESPACE::AttributeProto& proto, std::vector<int64_t>& values) {
    ORT_RETURN_IF_NOT(proto.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
    values.reserve(proto.ints_size());
    for (int i = 0; i < proto.ints_size(); i++) {
      values.push_back(proto.ints(i));
    }
    return Status::OK();
  }

  Status GetIntAttr(ONNX_NAMESPACE::AttributeProto& proto, int64_t& value) {
    ORT_RETURN_IF_NOT(proto.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    value = proto.i();
    return Status::OK();
  }

  Status GetFloatAttr(ONNX_NAMESPACE::AttributeProto& proto, float& value) {
    ORT_RETURN_IF_NOT(proto.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
    value = proto.f();
    return Status::OK();
  }
  Status GetStringAttr(ONNX_NAMESPACE::AttributeProto& proto, std::string& value) {
    ORT_RETURN_IF_NOT(proto.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
    value = proto.s();
    return Status::OK();
  }

  void InitDstReorderOutput(mkldnn::engine& cpu_engine,
                            mkldnn::memory::data_type& data_type,
                            std::vector<mkldnn::primitive>& net,
                            std::vector<std::unordered_map<int, mkldnn::memory>>& net_args);

  mkldnn::memory::format_tag GetSourceFormat(int dim_size);

 public:
  std::string name_;
  std::vector<std::shared_ptr<MklDnnKernel>> parents_;
  bool fuse_relu_ = false;
  bool fuse_sum_ = false;
  std::shared_ptr<mkldnn::memory> primitive_dst_mem_;
  std::unique_ptr<mkldnn::memory::desc> primitive_dst_md_;
  TensorShape primitive_dst_shape_;
  mkldnn::memory::desc primitive_dst_desc_;
  // ONNX Runtime format
  mkldnn::memory::format_tag ort_source_format_;
  mkldnn::memory::desc ort_source_desc_;

  // input format.
  // It can be ORT format (nchw) or blocked memory format from parent node
  // mkldnn::memory::format_tag source_format_ = mkldnn::memory::format_tag::any;
  mkldnn::memory::desc source_desc_ = mkldnn::memory::desc();
  Status primitive_created_status_;

 protected:
  // Pointer to MklNode of subgraph IR
  std::shared_ptr<MklDnnNode> mklnode_ptr_;
  // input format expected by primitive object
  mkldnn::memory::desc primitive_src_desc_;

  // memory used for reorders
  std::unique_ptr<mkldnn::memory> reorder_dst_mem_to_;
  AllocatorPtr alloc_;
  MKLDNNExecutionProvider* provider_;
};

}  // namespace mkl_dnn
}  // namespace onnxruntime
