// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"
#include "core/providers/mkldnn/mkldnn_common.h"
#include "core/providers/cpu/nn/autopad_type.h"
#include "core/providers/mkldnn/subgraph/mkl_kernel.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class MklSum : public MklKernel {
 public:
  explicit MklSum(MklNode& node,
                  MKLDNNExecutionProvider* provider,
                  std::shared_ptr<MKLContext> mkl_context) : MklKernel(node, provider, mkl_context) {
  }

  Status CreatePrimitives(const ONNXRunTimeTensor* input_tensors,
                          mkldnn::engine& cpu_engine,
                          std::vector<mkldnn::primitive>& net,
                          mkldnn::memory::format& source_format) override {
    int num_inputs = mklnode_ptr_->num_inputs;
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    std::vector<float> coeff;
    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.size() == 0) {
      auto xshape = input_tensors[input_index].shape;
      auto xdim = input_tensors[input_index].ndim;
      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
      source_format = ort_source_format_;
      src_format_ = ort_source_format_;
      x_shape = TensorShape(xshape, xdim);
    } else {
      x_shape = parents_[0].get()->primitive_dst_shape_;
      src_format_ = parents_[0].get()->primitive_dst_format_;
    }
    primitive_dst_shape_ = TensorShape(x_shape);

    mkldnn::memory::dims dst_dims_mkl(
        primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());

    for (int i = 0; i < num_inputs; i++) {
      TensorShape x_shape1;

      if (mklnode_ptr_->parent_nodes.size() == 0) {
        auto xshape = input_tensors[input_index].shape;
        auto xdim = input_tensors[input_index].ndim;
        mkldnn::memory::dims dims(xdim);
        ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
        x_shape1 = TensorShape(xshape, xdim);
        mkldnn::memory::dims src_dims_mkl(
            x_shape1.GetDims().begin(), x_shape1.GetDims().end());

        src_md_.reset(new mkldnn::memory::desc(
            {src_dims_mkl}, MklDnnType<T>(), src_format_));

        auto mpd = mkldnn::memory::primitive_desc(*src_md_, cpu_engine);
        auto src_memory = mkldnn::memory(mpd, nullptr);
        srcs_pd_.push_back(mpd);
        srcs_memory_.push_back(src_memory);
        coeff.push_back(1.0);
      } else {
        src_md_.reset(
            new mkldnn::memory::desc(parents_[i].get()->primitive_dst_mem_.get()->get_primitive_desc().desc()));
        auto mpd = mkldnn::memory::primitive_desc(*src_md_, cpu_engine);
        auto src_memory = *parents_[i].get()->primitive_dst_mem_;  //mkldnn::memory(mpd);
        srcs_pd_.push_back(mpd);
        srcs_memory_.push_back(src_memory);
        coeff.push_back(1.0);
        ort_source_format_ = source_format;
      }
    }

    primitive_dst_md_.reset(new mkldnn::memory::desc(
        {dst_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::any));
    sum_pd_.reset(new mkldnn::sum::primitive_desc(
        *primitive_dst_md_, coeff, srcs_pd_));
    primitive_dst_format_ = static_cast<mkldnn::memory::format>(sum_pd_->dst_primitive_desc().desc().data.format);

    if (mklnode_ptr_->output_index >= 0) {
      // last node of sub-graph. need to allocate memory for output_tensor
      if (primitive_dst_format_ != ort_source_format_) {
        // reorder neded. Use primitive output as input to reorder and
        // allocate buffer for reorder output, final output of this subgraph
        primitive_dst_mem_.reset(new mkldnn::memory(sum_pd_->dst_primitive_desc()));
      } else {
        // Last node but re-order not needed. Allocate buffer to output of this node
        primitive_dst_mem_.reset(new mkldnn::memory(sum_pd_->dst_primitive_desc(), nullptr));
      }
    } else {
      // Intermediate node. Use mkldnn kernel internal memory for output and
      // use this as input to next node.
      primitive_dst_mem_.reset(new mkldnn::memory(sum_pd_->dst_primitive_desc()));
    }
    primitive_dst_format_ = static_cast<mkldnn::memory::format>(sum_pd_->dst_primitive_desc().desc().data.format);

    std::vector<mkldnn::primitive::at> inputs;
    for (int i = 0; i < num_inputs; i++) {
      inputs.push_back(srcs_memory_[i]);
    }
    auto c = mkldnn::sum(*sum_pd_, inputs, *primitive_dst_mem_);
    net.push_back(c);

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      mkldnn::memory::data_type t = MklDnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net);
    }
    return Status::OK();
  }

  Status Compute(const ONNXRunTimeTensor* input_tensors,
                 ONNXRunTimeTensor* const output_tensors) override {
    int num_inputs = mklnode_ptr_->num_inputs;
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    std::vector<float> coeff;
    if (mklnode_ptr_->parent_nodes.size() == 0) {
      for (int i = 0; i < num_inputs; i++) {
        srcs_memory_[i].set_data_handle(input_tensors[input_index + i].data);
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      // Last node. Allocate output buffer memory and reorder if needed
      AllocateMemoryAndReorderIfNeeded(output_tensors, input_tensors[0].dtype);
    }
    return Status::OK();
  }

 private:
  std::unique_ptr<mkldnn::memory::desc> src_md_;
  std::vector<mkldnn::memory> srcs_memory_;

  std::vector<mkldnn::memory::primitive_desc> srcs_pd_;
  std::unique_ptr<mkldnn::memory::primitive_desc> src_mpd_;
  std::unique_ptr<mkldnn::memory::primitive_desc> dst_pd_;
  std::unique_ptr<mkldnn::sum::primitive_desc> sum_pd_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime
