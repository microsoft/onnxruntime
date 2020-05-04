// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {

template <typename T>
class DnnlRelu : public DnnlKernel {
 public:
  DnnlRelu(const DnnlNode& node,
           DNNLExecutionProvider* provider,
           const Provider_NodeAttributes& attributes,
           const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ORT_UNUSED_PARAMETER(attributes);
    ORT_UNUSED_PARAMETER(attributes_prefix);
  }

  void CreatePrimitives(const OrtCustomOpApi* api,
                        OrtKernelContext* context,
                        dnnl::engine& cpu_engine,
                        std::vector<dnnl::primitive>& net,
                        std::vector<std::unordered_map<int, dnnl::memory>>& net_args) {
    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

      auto xshape = tensor_shape.data();
      auto xdim = tensor_shape.size();

      dnnl::memory::dims dims(xdim);

      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));

      x_shape = TensorShape(xshape, xdim);

      if (x_shape.NumDimensions() == 0) {
        primitive_created_status_ = Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Shape of size zero " + x_shape.ToString());
        return;
      }

      dnnl::memory::dims src_dims(
          x_shape.GetDims().begin(), x_shape.GetDims().end());

      ort_source_desc_ = dnnl::memory::desc(
          {src_dims}, DnnnType<T>(), ort_source_format_);
      source_desc_ = ort_source_desc_;
      src_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc({src_dims}, DnnnType<T>(), ort_source_format_));
      src_mem_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory({{src_dims}, DnnnType<T>(), ort_source_format_}, cpu_engine, nullptr));
    } else {
      src_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc(parents_[0].get()->primitive_dst_desc_));
      src_mem_ = parents_[0].get()->primitive_dst_mem_;
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;
    }

    primitive_dst_shape_ = TensorShape(x_shape);

    dnnl::memory::dims dst_dims_mkl(primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());
    dnnl::algorithm algo = dnnl::algorithm::eltwise_relu;
    fwd_desc_ = onnxruntime::make_unique<dnnl::eltwise_forward::desc>(
        dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algo, *src_md_, 0));
    relu_fwd_pd_ = onnxruntime::make_unique<dnnl::eltwise_forward::primitive_desc>(
        dnnl::eltwise_forward::primitive_desc(*fwd_desc_, cpu_engine));

    primitive_src_desc_ = relu_fwd_pd_.get()->src_desc();
    primitive_dst_desc_ = relu_fwd_pd_.get()->dst_desc();

    if (mklnode_ptr_->output_index >= 0) {
      // last node of sub-graph. need to allocate memory for output_tensor
      if (primitive_dst_desc_ != ort_source_desc_) {
        // reorder neded. Use primitive output as input to reorder and
        // allocate buffer for reorder output, final output of this subgraph
        primitive_dst_mem_ = std::make_shared<dnnl::memory>(dnnl::memory(relu_fwd_pd_.get()->dst_desc(), cpu_engine));
      } else {
        // Last node but re-order not needed. Allocate buffer to output of this node
        primitive_dst_mem_ = std::make_shared<dnnl::memory>(dnnl::memory(relu_fwd_pd_.get()->dst_desc(), cpu_engine, nullptr));
      }
    } else {
      // Intermediate node. Use dnnl kernel internal memory for output and
      // use this as input to next node.
      primitive_dst_mem_ = std::make_shared<dnnl::memory>(dnnl::memory(relu_fwd_pd_.get()->dst_desc(), cpu_engine));
    }

    relu_fwd_ = onnxruntime::make_unique<dnnl::eltwise_forward>(
        dnnl::eltwise_forward(*relu_fwd_pd_));

    net.push_back(*relu_fwd_);

    net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                        {DNNL_ARG_DST, *primitive_dst_mem_}});

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      dnnl::memory::data_type t = DnnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net, net_args);
    }
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};

    ORT_RETURN_IF_ERROR(primitive_created_status_);

    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (mklnode_ptr_->parent_nodes.empty()) {
      // Sub-graph's first node. Read input from input buffer
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
      src_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
    }

    if (mklnode_ptr_->output_index >= 0) {
      auto& y_dims = primitive_dst_shape_.GetDims();
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
  std::shared_ptr<dnnl::memory> src_mem_;

  std::unique_ptr<dnnl::eltwise_forward::desc> fwd_desc_;
  std::unique_ptr<dnnl::eltwise_forward::primitive_desc> relu_fwd_pd_;
  std::unique_ptr<dnnl::primitive> relu_fwd_;

  std::unique_ptr<dnnl::memory::desc> src_md_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
