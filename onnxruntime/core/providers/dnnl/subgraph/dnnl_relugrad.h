// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"
#include "core/providers/dnnl/subgraph/dnnl_activations.h"

/*
x_grad   +----------------+
-------->+                |
         | ReluGrad       | primitive_dst
         |  or            +---------->
x        | Backward Relu  |
-------->+                |
         +----------------+

The ReluGrad operator has two tensors that come from the ONNXRuntime.
These are converted to dnnl memory descrptors 

The output parameter name is inherited from the DnnlKernel the name primitive_dst
name was chosen to match with the dnnl name when running in inference mode. The
dst for destination matches the naming scheme for dnnl in inference mode.  When
in training mode primitive_src would make more since but we are reusing the
already provided output parameter.

The dnnl backward primitive description requires the forward primitive
description. Since forward primitive description requires the memory
description (md) from the forward Relu operator. Since the actual memory
description from forward Relu is not avalible we are using the memory
description from the x input. With the belive that the important parts of
the memory descrption are the shape and dimentions which match the x input.
*/

namespace onnxruntime {
namespace ort_dnnl {
template <typename T>
class DnnlReluGrad : public DnnlKernel {
 public:
  DnnlReluGrad(const DnnlNode& node,
               DNNLExecutionProvider* provider,
               const NodeAttributes& attributes,
               const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ORT_UNUSED_PARAMETER(attributes);
    ORT_UNUSED_PARAMETER(attributes_prefix);
  }

  void AddForwardDnnlKernel(std::shared_ptr<DnnlRelu<T>> relu_fwd) {
    relu_fwd_ = relu_fwd;
  }

  void CreatePrimitives(const OrtCustomOpApi* api,
                        OrtKernelContext* context,
                        const std::unordered_map<dnnl::engine::kind, dnnl::engine>& dnnl_engine,
                        std::vector<dnnl::primitive>& net,
                        std::vector<std::unordered_map<int, dnnl::memory>>& net_args) override {
    dnnl::engine cpu_engine;
    dnnl::engine engine_to_use;
    std::unordered_map<dnnl::engine::kind, dnnl::engine>::const_iterator iter = dnnl_engine.find(dnnl::engine::kind::cpu);
    if (iter != dnnl_engine.end()) {
      cpu_engine = iter->second;
      engine_to_use = cpu_engine;
    }
    gpu_available_ = false;
    dnnl::engine gpu_engine;
    iter = dnnl_engine.find(dnnl::engine::kind::gpu);
    if (iter != dnnl_engine.end()) {
      gpu_engine = (dnnl::engine)(iter->second);
      gpu_available_ = true;
      engine_to_use = gpu_engine;
      LOGS_DEFAULT(INFO) << "gpu engine found" << std::endl;
    }

    if (!relu_fwd_) {
      ORT_THROW(
          "Unable to find forward pass Relu opt. "
          "Verify AddForwardDnnlKernel was called after DnnlReluGrad was created.");
    }

    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    TensorShape xgrad_shape;
    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.empty()) {
      // convert the onnx gradient Tensor to dnnl::memory and dnnl::memory::desc
      const OrtValue* xgrad_input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto xgrad_tensor_info = ort.GetTensorTypeAndShape(xgrad_input_tensor);
      auto xgrad_tensor_shape = ort.GetTensorShape(xgrad_tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(xgrad_tensor_info);

      auto xgradshape = xgrad_tensor_shape.data();
      auto xgraddim = xgrad_tensor_shape.size();

      dnnl::memory::dims xgraddims(xgraddim);

      ort_source_format_ = GetSourceFormat(static_cast<int>(xgraddim));

      xgrad_shape = TensorShape(xgradshape, xgraddim);

      if (xgrad_shape.Size() == 0) {
        primitive_created_status_ = Status(common::ONNXRUNTIME,
                                           common::INVALID_ARGUMENT,
                                           "Shape of size zero " + xgrad_shape.ToString());
        return;
      }

      dnnl::memory::dims xgrad_dims(
          xgrad_shape.GetDims().begin(), xgrad_shape.GetDims().end());

      ort_source_desc_ = dnnl::memory::desc(
          {xgrad_dims}, DnnnType<T>(), ort_source_format_);
      source_desc_ = ort_source_desc_;
      diff_dst_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc({xgrad_dims}, DnnnType<T>(), ort_source_format_));
      diff_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory({{xgrad_dims}, DnnnType<T>(), ort_source_format_}, cpu_engine, nullptr));

      // convert the onnx Tensor to dnnl::memory and dnnl::memory::desc
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

      auto xshape = tensor_shape.data();
      auto xdim = tensor_shape.size();

      dnnl::memory::dims dims(xdim);

      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));

      x_shape = TensorShape(xshape, xdim);

      if (x_shape.Size() == 0) {
        primitive_created_status_ = Status(common::ONNXRUNTIME,
                                           common::INVALID_ARGUMENT,
                                           "Shape of size zero " + x_shape.ToString());
        return;
      }

      dnnl::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());

      ort_source_desc_ = dnnl::memory::desc({src_dims}, DnnnType<T>(), ort_source_format_);
      source_desc_ = ort_source_desc_;
      src_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc({src_dims}, DnnnType<T>(), ort_source_format_));
      src_mem_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory({{src_dims}, DnnnType<T>(), ort_source_format_}, cpu_engine, nullptr));
      if (gpu_available_) {
        src_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(*src_md_, gpu_engine);
        net.push_back(mkldnn::reorder(*src_mem_, *src_mem_gpu_));
        net_args.push_back({{MKLDNN_ARG_SRC, *src_mem_},
                            {MKLDNN_ARG_DST, *src_mem_gpu_}});
        diff_dst_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(*diff_dst_md_, gpu_engine);
        net.push_back(mkldnn::reorder(*diff_dst_mem_, *diff_dst_mem_gpu_));
        net_args.push_back({{MKLDNN_ARG_SRC, *diff_dst_mem_},
                            {MKLDNN_ARG_DST, *diff_dst_mem_gpu_}});
      }
    } else {
      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;

      diff_dst_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc(parents_[0].get()->primitive_dst_desc_));
      src_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc(parents_[1].get()->primitive_dst_desc_));

      if (!gpu_available_) {
        diff_dst_mem_ = parents_[0].get()->primitive_dst_mem_;
        src_mem_ = parents_[1].get()->primitive_dst_mem_;
      } else {
        diff_dst_mem_gpu_ = parents_[0].get()->primitive_dst_mem_;
        src_mem_gpu_ = parents_[1].get()->primitive_dst_mem_;
      }

      xgrad_shape = parents_[0].get()->primitive_dst_shape_;
      x_shape = parents_[1].get()->primitive_dst_shape_;
    }

    primitive_dst_shape_ = TensorShape(x_shape);

    dnnl::memory::dims dst_dims_mkl(primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());
    dnnl::algorithm algo = dnnl::algorithm::eltwise_relu;

    relu_bwd_desc_ = onnxruntime::make_unique<dnnl::eltwise_backward::desc>(
        dnnl::eltwise_backward::desc(algo, *diff_dst_md_, *src_md_, 0.0, 0.0));
    relu_bwd_pd_ = onnxruntime::make_unique<dnnl::eltwise_backward::primitive_desc>(
        dnnl::eltwise_backward::primitive_desc(*relu_bwd_desc_, engine_to_use, *(relu_fwd_->GetPrimitiveDesc())));

    primitive_src_desc_ = relu_bwd_pd_.get()->src_desc();
    primitive_dst_desc_ = relu_bwd_pd_.get()->diff_src_desc();

    if (!gpu_available_) {
      if (mklnode_ptr_->output_index >= 0) {
        // last node of sub-graph. need to allocate memory for output_tensor
        if (primitive_dst_desc_ != ort_source_desc_) {
          // reorder neded. Use primitive output as input to reorder and
          // allocate buffer for reorder output, final output of this subgraph
          primitive_dst_mem_ = std::make_shared<dnnl::memory>(dnnl::memory(relu_bwd_pd_.get()->diff_src_desc(), cpu_engine));
        } else {
          // Last node but re-order not needed. Allocate buffer to output of this node
          primitive_dst_mem_ = std::make_shared<dnnl::memory>(dnnl::memory(relu_bwd_pd_.get()->diff_src_desc(), cpu_engine, nullptr));
        }
      } else {
        // Intermediate node. Use dnnl kernel internal memory for output and
        // use this as input to next node.
        primitive_dst_mem_ = std::make_shared<dnnl::memory>(dnnl::memory(relu_bwd_pd_.get()->diff_src_desc(), cpu_engine));
      }
    } else {
      primitive_dst_mem_ = std::make_shared<dnnl::memory>(dnnl::memory(relu_bwd_pd_.get()->diff_src_desc(), gpu_engine));
    }

    net.push_back(dnnl::eltwise_backward(*relu_bwd_pd_));

    if (!gpu_available_) {
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                          {DNNL_ARG_DIFF_DST, *diff_dst_mem_},
                          {DNNL_ARG_DIFF_SRC, *primitive_dst_mem_}});
    } else {
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_gpu_},
                          {DNNL_ARG_DIFF_DST, *diff_dst_mem_gpu_},
                          {DNNL_ARG_DIFF_SRC, *primitive_dst_mem_}});
    }

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      dnnl::memory::data_type t = DnnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net, net_args, gpu_available_);
    }
  }

  virtual Status Bind(const OrtCustomOpApi* api,
                      OrtKernelContext* context) {
    Ort::CustomOpApi ort{*api};

    ORT_RETURN_IF_ERROR(primitive_created_status_);

    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* dx_input_tensor = ort.KernelContext_GetInput(context, input_index);
      const T* dx_data = const_cast<T*>(ort.GetTensorData<T>(dx_input_tensor));
      diff_dst_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(dx_data)));
      // Sub-graph's first node. Read input from input buffer
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
      const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
      src_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
    }

    if (mklnode_ptr_->output_index >= 0) {
      auto& y_dims = primitive_dst_shape_.GetDims();
      // Allocate memory for output bufffer
      OrtValue* output = ort.KernelContext_GetOutput(context, mklnode_ptr_->output_index, &y_dims[0], static_cast<int>(primitive_dst_shape_.GetDims().size()));
      T* dst_data = ort.GetTensorMutableData<T>(output);

      if (!gpu_available_) {
        if (primitive_dst_desc_ != ort_source_desc_) {
          reorder_dst_mem_to_->set_data_handle(dst_data);
        } else {
          primitive_dst_mem_->set_data_handle(dst_data);
        }
      } else {  // gpu_available_
        reorder_dst_mem_to_->set_data_handle(dst_data);
      }
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<DnnlRelu<T>> relu_fwd_;
  std::shared_ptr<dnnl::memory> diff_dst_mem_;
  std::shared_ptr<dnnl::memory> diff_dst_mem_gpu_;

  std::unique_ptr<dnnl::memory::desc> diff_dst_md_;

  std::shared_ptr<dnnl::memory> src_mem_;
  std::shared_ptr<dnnl::memory> src_mem_gpu_;

  std::unique_ptr<dnnl::memory::desc> src_md_;

  std::unique_ptr<dnnl::eltwise_backward::desc> relu_bwd_desc_;
  std::unique_ptr<dnnl::eltwise_backward::primitive_desc> relu_bwd_pd_;

  bool gpu_available_;
};  // namespace ort_dnnl
}  // namespace ort_dnnl
}  // namespace onnxruntime