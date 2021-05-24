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
                   const NodeAttributes& attributes,
                   const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ReadAttributes(attributes, attributes_prefix);
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
      cpu_engine = (dnnl::engine)iter->second;
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
        if (gpu_available_) {
          auto src_memory_gpu = dnnl::memory(mpd, gpu_engine);
          net.push_back(dnnl::reorder(src_memory, src_memory_gpu));
          net_args.push_back({{DNNL_ARG_SRC, src_memory},
                              {DNNL_ARG_DST, src_memory_gpu}});
          srcs_memory_gpu_.push_back(src_memory_gpu);
        }
        srcs_pd_.push_back(mpd);
        srcs_memory_.push_back(src_memory);
        coeff.push_back(1.0);
      } else {
        x_shape = parents_[0].get()->primitive_dst_shape_;
        auto mpd = dnnl::memory::desc(parents_[i].get()->primitive_dst_desc_);
        if (!gpu_available_) {
          auto src_memory = *parents_[i].get()->primitive_dst_mem_;
          srcs_memory_.push_back(src_memory);
        } else {  // gpu_available_
          auto src_memory_gpu = *parents_[i].get()->primitive_dst_mem_;
          srcs_memory_gpu_.push_back(src_memory_gpu);
        }
        srcs_pd_.push_back(mpd);
        coeff.push_back(1.0);
      }
    }

    primitive_dst_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({dst_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));
    if (!gpu_available_) {
      sum_pd_ = std::make_unique<dnnl::sum::primitive_desc>(
          dnnl::sum::primitive_desc(*primitive_dst_md_, coeff, srcs_pd_, cpu_engine));
    } else {  // gpu_available_
      sum_pd_ = std::make_unique<dnnl::sum::primitive_desc>(
          dnnl::sum::primitive_desc(*primitive_dst_md_, coeff, srcs_pd_, gpu_engine));
    }

    if (!gpu_available_) {
      if (mklnode_ptr_->output_index >= 0) {
        // last node of sub-graph. need to allocate memory for output_tensor
        if (primitive_dst_desc_ != ort_source_desc_) {
          // reorder neded. Use primitive output as input to reorder and
          // allocate buffer for reorder output, final output of this subgraph
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(sum_pd_->dst_desc(), cpu_engine));
        } else {
          // Last node but re-order not needed. Allocate buffer to output of this node
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(sum_pd_->dst_desc(), cpu_engine, nullptr));
        }
      } else {
        // Intermediate node. Use Dnnl kernel internal memory for output and
        // use this as input to next node.
        primitive_dst_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(sum_pd_->dst_desc(), cpu_engine));
      }
    } else {  // gpu_available_
      primitive_dst_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(sum_pd_->dst_desc(), gpu_engine));
    }
    primitive_dst_desc_ = sum_pd_->dst_desc();

    auto c = dnnl::sum(*sum_pd_);
    net.push_back(c);
    std::unordered_map<int, dnnl::memory> args;
    if (!gpu_available_) {
      args.insert({DNNL_ARG_DST, *primitive_dst_mem_});
      for (int i = 0; i < (int)num_inputs; i++) {
        args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs_memory_[i]});
      }
    } else {  // gpu_available_
      args.insert({DNNL_ARG_DST, *primitive_dst_mem_});
      for (int i = 0; i < (int)num_inputs; i++) {
        args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs_memory_gpu_[i]});
      }
    }
    net_args.push_back(args);

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      dnnl::memory::data_type t = DnnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net, net_args, gpu_available_);
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
  std::unique_ptr<dnnl::memory::desc> src_md_;
  std::vector<dnnl::memory::desc> src_mds_;
  std::vector<dnnl::memory> srcs_memory_;
  std::vector<dnnl::memory> srcs_memory_gpu_;

  std::vector<dnnl::memory::desc> srcs_pd_;
  std::unique_ptr<dnnl::memory::desc> src_mpd_;
  std::unique_ptr<dnnl::memory::desc> dst_pd_;
  std::unique_ptr<dnnl::sum::primitive_desc> sum_pd_;

  bool gpu_available_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
