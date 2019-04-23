// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/op_kernel.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"
#include "core/providers/cpu/nn/autopad_type.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#include "core/providers/mkldnn/subgraph/mkl_kernel.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class MklLrn : public MklKernel {
 public:
  MklLrn(MklNode& node,
         MKLDNNExecutionProvider* provider,
         std::shared_ptr<MKLContext> mkl_context) : MklKernel(node, provider, mkl_context) {
  }

  Status CreatePrimitives(const ONNXRunTimeTensor* input_tensors,
                          mkldnn::engine& cpu_engine,
                          std::vector<mkldnn::primitive>& net,
                          mkldnn::memory::format& source_format) {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.size() == 0) {
      auto xshape = input_tensors[input_index].shape;
      auto xdim = input_tensors[input_index].ndim;
      mkldnn::memory::dims dims(xdim);
      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
      source_format = ort_source_format_;
      x_shape = TensorShape(xshape, xdim);

      mkldnn::memory::dims src_dims_mkl(
          x_shape.GetDims().begin(), x_shape.GetDims().end());

      src_md_.reset(new mkldnn::memory::desc(
          {src_dims_mkl}, MklDnnType<T>(), source_format));
      src_mem_.reset(
          new mkldnn::memory({*src_md_, cpu_engine}, nullptr));
    } else {
      src_md_.reset(
          new mkldnn::memory::desc(parents_[0].get()->primitive_dst_mem_.get()->get_primitive_desc().desc()));
      src_mem_ = parents_[0].get()->primitive_dst_mem_;
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = source_format;
    }

    primitive_dst_shape_ = TensorShape(x_shape);

    mkldnn::algorithm algo = mkldnn::algorithm::lrn_across_channels;
    fwd_desc_.reset(new mkldnn::lrn_forward::desc(
        mkldnn::prop_kind::forward_scoring, algo, *src_md_,
        size_, alpha_, beta_, bias_));

    fwd_primitive_desc_.reset(new mkldnn::lrn_forward::primitive_desc(
        *fwd_desc_, cpu_engine));

    primitive_src_format_ = static_cast<mkldnn::memory::format>(
        fwd_primitive_desc_.get()->src_primitive_desc().desc().data.format);
    primitive_dst_format_ = static_cast<mkldnn::memory::format>(
        fwd_primitive_desc_.get()->dst_primitive_desc().desc().data.format);

    if (mklnode_ptr_->output_index >= 0) {
      // last node of sub-graph. need to allocate memory for output_tensor
      if (primitive_dst_format_ != ort_source_format_) {
        // reorder neded. Use primitive output as input to reorder and
        // allocate buffer for reorder output, final output of this subgraph
        primitive_dst_mem_.reset(
            new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc()));
      } else {
        // Last node but re-order not needed. Allocate buffer to output of this node
        primitive_dst_mem_.reset(
            new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc(), nullptr));
      }
    } else {
      // Intermediate node. Use mkldnn kernel internal memory for output and
      // use this as input to next node.
      primitive_dst_mem_.reset(
          new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc()));
    }

    lrn_fwd_.reset(
        new mkldnn::lrn_forward(*fwd_primitive_desc_, *src_mem_, *primitive_dst_mem_));
    net.push_back(*lrn_fwd_);

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      mkldnn::memory::data_type t = MklDnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net);
    }

    return Status::OK();
  }

  Status Bind(const ONNXRunTimeTensor* input_tensors,
                 ONNXRunTimeTensor* const output_tensors) override {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (mklnode_ptr_->parent_nodes.size() == 0) {
      // Sub-graph's first node. Read input from input buffer
      src_mem_->set_data_handle(input_tensors[input_index].data);
    }

    if (mklnode_ptr_->output_index >= 0) {
      AllocateMemoryAndReorderIfNeeded(output_tensors, input_tensors[0].dtype);
    }

    return Status::OK();
  }
  void ReadAttributes(const std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto>& attributes,
                      const std::string attributes_prefix = "") override {
    auto attr = attributes.find(attributes_prefix + "size");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
      size_ = attr->second.i();
    }
    ORT_ENFORCE(size_ > 0);
    ORT_ENFORCE(size_ % 2 == 1);

    attr = attributes.find(attributes_prefix + "alpha");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      alpha_ = attr->second.f();
    }

    attr = attributes.find(attributes_prefix + "beta");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      beta_ = attr->second.f();
    }

    bias_ = 1.0f;
    attr = attributes.find(attributes_prefix + "bias");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      bias_ = attr->second.f();
    }
  }

 private:
  float alpha_ = 0;
  float beta_ = 0;
  float bias_ = 0;
  int size_ = 0;

 private:
  std::shared_ptr<mkldnn::memory> src_mem_;

  std::unique_ptr<mkldnn::lrn_forward::desc> fwd_desc_;
  std::unique_ptr<mkldnn::lrn_forward::primitive_desc> fwd_primitive_desc_;
  std::unique_ptr<mkldnn::primitive> lrn_fwd_;

  std::unique_ptr<mkldnn::memory::desc> src_md_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
