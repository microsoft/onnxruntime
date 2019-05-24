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
  explicit MklDnnKernel(MklDnnNode& node,
                        MKLDNNExecutionProvider* provider) {
    mklnode_ptr_ = std::make_shared<MklDnnNode>(node);
    provider_ = provider;
    alloc_ = provider_->GetAllocator(0, OrtMemTypeDefault);
  }
  virtual ~MklDnnKernel(){};

  virtual Status CreatePrimitives(Ort::CustomOpApi ort,
                                  OrtKernelContext* context,
                                  mkldnn::engine& cpu_engine,
                                  std::vector<mkldnn::primitive>& net,
                                  mkldnn::memory::format& src_fmt) = 0;

  virtual void ReorderWeights(Ort::CustomOpApi ort, OrtKernelContext* context, mkldnn::engine& cpu_engine) {
    ORT_UNUSED_PARAMETER(ort);
    ORT_UNUSED_PARAMETER(cpu_engine);
  }

  virtual Status Bind(Ort::CustomOpApi ort, OrtKernelContext* context) = 0;

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
                            std::vector<mkldnn::primitive>& net);

  //void AllocateMemoryAndReorderIfNeeded(const OrtCustomOpApi* api);

  //void AllocateOutputTensor(const OrtCustomOpApi* api, int index, const int64_t* shape, size_t dim);

  mkldnn::memory::format GetSourceFormat(int dim_size);

 public:
  std::vector<std::shared_ptr<MklDnnKernel>> parents_;
  bool fuse_relu_ = false;
  bool fuse_sum_ = false;
  std::shared_ptr<mkldnn::memory> primitive_dst_mem_;
  std::unique_ptr<mkldnn::memory::desc> primitive_dst_md_;
  TensorShape primitive_dst_shape_;
  mkldnn::memory::format primitive_dst_format_ = mkldnn::memory::format::any;

 protected:
  // ONNX Runtime format
  mkldnn::memory::format ort_source_format_ = mkldnn::memory::format::any;
  // input format.
  // It can be ORT format (nchw) or blocked memory format from parent nce
  mkldnn::memory::format src_format_ = mkldnn::memory::format::any;
  // Pointer to MklNode of subgraph IR
  std::shared_ptr<MklDnnNode> mklnode_ptr_;
  // input format expected by primitive object
  mkldnn::memory::format primitive_src_format_ = mkldnn::memory::format::any;

  // memory used for reorders
  std::unique_ptr<mkldnn::memory> reorder_dst_mem_to_;

 protected:
  Status primitive_created_;
  AllocatorPtr alloc_;
  MKLDNNExecutionProvider* provider_;
};

}  // namespace mkl_dnn
}  // namespace onnxruntime
