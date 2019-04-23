// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "mkldnn.hpp"
#include "core/common/cpuid_info.h"
#include "core/providers/mkldnn/subgraph/subgraph.h"

namespace onnxruntime {
namespace mkl_dnn {

class MklKernel {
 public:
  explicit MklKernel(MklNode& node,
                     MKLDNNExecutionProvider* provider,
                     std::shared_ptr<MKLContext> mkl_context) {
    mkl_context_ = mkl_context;
    mklnode_ptr_ = std::make_shared<MklNode>(node);
    provider_ = provider;
    alloc_ = provider_->GetAllocator(0, OrtMemTypeDefault);
  }
  virtual ~MklKernel(){};

  virtual void ReadAttributes(const std::unordered_map<std::string,
                                                       ONNX_NAMESPACE::AttributeProto>& attributes,
                              const std::string attributes_prefix = "") {
    ORT_UNUSED_PARAMETER(attributes);
    ORT_UNUSED_PARAMETER(attributes_prefix);
  }

  virtual Status CreatePrimitives(const ONNXRunTimeTensor* input_tensors,
                                  mkldnn::engine& cpu_engine,
                                  std::vector<mkldnn::primitive>& net,
                                  mkldnn::memory::format& src_fmt) = 0;

  virtual void ReorderWeights(const ONNXRunTimeTensor* input_tensors, mkldnn::engine& cpu_engine) {
    ORT_UNUSED_PARAMETER(input_tensors);
    ORT_UNUSED_PARAMETER(cpu_engine);
  }

  virtual Status Submit(const ONNXRunTimeTensor* input_tensors,
                         ONNXRunTimeTensor* const output_tensors) = 0;

 protected:
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
                            std::vector<mkldnn::primitive>& net) {
    // Allocate dst buffer if reorder is necessary
    if (primitive_dst_format_ != ort_source_format_) {
      // reorder to ONNXRuntime format
      mkldnn::memory::dims dst_dims_mkl(
          primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());
      mkldnn::memory::desc dst_des = mkldnn::memory::desc(dst_dims_mkl,
                                                          data_type, ort_source_format_);
      reorder_dst_mem_to_.reset(new mkldnn::memory({dst_des, cpu_engine}, nullptr));
      net.push_back(mkldnn::reorder(*primitive_dst_mem_, *reorder_dst_mem_to_));
    }
  }

  void AllocateMemoryAndReorderIfNeeded(ONNXRunTimeTensor* const output_tensors, const DType& dtype) {
    // End of sub-graph. Allocate memory and get the output
    auto& y_dims = primitive_dst_shape_.GetDims();
    AllocateOutputTensor(output_tensors, mklnode_ptr_->output_index,
                          &y_dims[0], static_cast<int>(primitive_dst_shape_.GetDims().size()),
                          dtype);
    if (primitive_dst_format_ != ort_source_format_) {
      reorder_dst_mem_to_->set_data_handle(output_tensors[mklnode_ptr_->output_index].data);
    } else {
      primitive_dst_mem_->set_data_handle(output_tensors[mklnode_ptr_->output_index].data);
    }
  }

  void AllocateOutputTensor(ONNXRunTimeTensor* const output_tensors, int index, const int64_t* shape, size_t dim, const DType& dtype) {
    output_tensors[index].dtype = dtype;
    output_tensors[index].ndim = dim;
    output_tensors[index].shape = new int64_t[dim];
    memcpy(output_tensors[index].shape, shape, sizeof(int64_t) * dim);
    int64_t data_size = 1;
    for (auto j = 0; j < output_tensors[index].ndim; j++)
      data_size *= output_tensors[index].shape[j];
    output_tensors[index].data = (*(mkl_context_->allocate_func))(mkl_context_->allocator, sizeof(double) * data_size, 64);
  }

  mkldnn::memory::format GetSourceFormat(int dim_size) {
    mkldnn::memory::format source_format = mkldnn::memory::format::any;
    switch (dim_size) {
      case 1: {
        source_format = mkldnn::memory::format::x;
        break;
      }
      case 2: {
        source_format = mkldnn::memory::format::nc;
        break;
      }
      case 3: {
        source_format = mkldnn::memory::format::ntc;
        break;
      }
      case 4: {
        source_format = mkldnn::memory::format::nchw;
        break;
      }
      case 5: {
        source_format = mkldnn::memory::format::ncdhw;
        break;
      }
      default: {
        source_format = mkldnn::memory::format::any;
        break;
      }
    }

    return source_format;
  }

 public:
  std::vector<std::shared_ptr<MklKernel>> parents_;
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
  std::shared_ptr<MklNode> mklnode_ptr_;
  // input format expected by primitive object
  mkldnn::memory::format primitive_src_format_ = mkldnn::memory::format::any;

  // memory used for reorders
  std::unique_ptr<mkldnn::memory> reorder_dst_mem_to_;

 protected:
  Status primitive_created_;
  std::shared_ptr<MKLContext> mkl_context_;

  AllocatorPtr alloc_;
  MKLDNNExecutionProvider* provider_;
};

}  // namespace mkl_dnn
}  // namespace onnxruntime
