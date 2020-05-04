// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_common.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {

template <typename T>
class DnnlSum : public DnnlKernel {
 public:
  explicit DnnlSum(const DnnlNode& node,
                   DNNLExecutionProvider* provider,
                   const Provider_NodeAttributes& attributes,
                   const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ReadAttributes(attributes, attributes_prefix);
  }

  void CreatePrimitives(const OrtCustomOpApi* api,
                        OrtKernelContext* context,
                        dnnl::engine& cpu_engine,
                        std::vector<dnnl::primitive>& net,
                        std::vector<std::unordered_map<int, dnnl::memory>>& net_args) override {
    Ort::CustomOpApi ort{*api};
    int num_inputs = mklnode_ptr_->num_inputs;
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    std::vector<float> coeff;
    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      auto xshape = tensor_shape.data();
      auto xdim = tensor_shape.size();

      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));

      x_shape = TensorShape(xshape, xdim);
      dnnl::memory::dims src_dims(
          x_shape.GetDims().begin(), x_shape.GetDims().end());
      ort_source_desc_ = dnnl::memory::desc(
          {src_dims}, DnnnType<T>(), ort_source_format_);
      source_desc_ = ort_source_desc_;
    } else {
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;
    }
    primitive_dst_shape_ = TensorShape(x_shape);

    dnnl::memory::dims dst_dims_mkl(
        primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());

    for (int i = 0; i < num_inputs; i++) {
      TensorShape x_shape1;

      if (mklnode_ptr_->parent_nodes.empty()) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        auto tensor_shape = ort.GetTensorShape(tensor_info);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
        auto xshape = tensor_shape.data();
        auto xdim = tensor_shape.size();
        dnnl::memory::dims dims(xdim);

        x_shape1 = TensorShape(xshape, xdim);
        dnnl::memory::dims src_dims(
            x_shape1.GetDims().begin(), x_shape1.GetDims().end());
        auto mpd = dnnl::memory::desc({src_dims}, DnnnType<T>(), ort_source_format_);
        auto src_memory = dnnl::memory(mpd, cpu_engine, nullptr);
        srcs_pd_.push_back(mpd);
        srcs_memory_.push_back(src_memory);
        coeff.push_back(1.0);
      } else {
        x_shape = parents_[0].get()->primitive_dst_shape_;
        auto mpd = dnnl::memory::desc(parents_[i].get()->primitive_dst_desc_);
        auto src_memory = *parents_[i].get()->primitive_dst_mem_;
        srcs_pd_.push_back(mpd);
        srcs_memory_.push_back(src_memory);
        coeff.push_back(1.0);
      }
    }

    primitive_dst_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({dst_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));
    sum_pd_ = onnxruntime::make_unique<dnnl::sum::primitive_desc>(
        dnnl::sum::primitive_desc(*primitive_dst_md_, coeff, srcs_pd_, cpu_engine));

    if (mklnode_ptr_->output_index >= 0) {
      // last node of sub-graph. need to allocate memory for output_tensor
      if (primitive_dst_desc_ != ort_source_desc_) {
        // reorder neded. Use primitive output as input to reorder and
        // allocate buffer for reorder output, final output of this subgraph
        primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
            dnnl::memory(sum_pd_->dst_desc(), cpu_engine));
      } else {
        // Last node but re-order not needed. Allocate buffer to output of this node
        primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
            dnnl::memory(sum_pd_->dst_desc(), cpu_engine, nullptr));
      }
    } else {
      // Intermediate node. Use Dnnl kernel internal memory for output and
      // use this as input to next node.
      primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory(sum_pd_->dst_desc(), cpu_engine));
    }
    primitive_dst_desc_ = sum_pd_->dst_desc();

    auto c = dnnl::sum(*sum_pd_);
    net.push_back(c);
    std::unordered_map<int, dnnl::memory> args{
        {DNNL_ARG_DST, *primitive_dst_mem_}};
    for (int i = 0; i < (int)num_inputs; i++) {
      args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs_memory_[i]});
    }
    net_args.push_back(args);

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      dnnl::memory::data_type t = DnnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net, net_args);
    }
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};

    int num_inputs = mklnode_ptr_->num_inputs;
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (mklnode_ptr_->parent_nodes.empty()) {
      for (int i = 0; i < num_inputs; i++) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index + i);
        const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
        srcs_memory_[i].set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      // Last node. Allocate output buffer memory and reorder if needed
      const auto& y_dims = primitive_dst_shape_.GetDims();
      // Allocate memory for output bufffer
      OrtValue* output = ort.KernelContext_GetOutput(context, mklnode_ptr_->output_index, &y_dims[0], static_cast<int>(primitive_dst_shape_.GetDims().size()));
      T* dst_data = ort.GetTensorMutableData<T>(output);

      if (primitive_dst_desc_ != ort_source_desc_) {
        reorder_dst_mem_to_->set_data_handle(dst_data);
      } else {
        primitive_dst_mem_->set_data_handle(dst_data);
      }
    }
    return Status::OK();
  }

 private:
  std::unique_ptr<dnnl::memory::desc> src_md_;
  std::vector<dnnl::memory::desc> src_mds_;
  std::vector<dnnl::memory> srcs_memory_;

  std::vector<dnnl::memory::desc> srcs_pd_;
  std::unique_ptr<dnnl::memory::desc> src_mpd_;
  std::unique_ptr<dnnl::memory::desc> dst_pd_;
  std::unique_ptr<dnnl::sum::primitive_desc> sum_pd_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
