// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {

template <typename T>
class DnnlMatmul : public DnnlKernel {
 public:
  DnnlMatmul(const DnnlNode& node,
           DNNLExecutionProvider* provider,
           const NodeAttributes& attributes,
           const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ReadAttributes(attributes, attributes_prefix);
  }

  void CreatePrimitives(const OrtCustomOpApi* api,
                        OrtKernelContext* context,
                        const std::unordered_map<dnnl::engine::kind, dnnl::engine>& dnnl_engine,
                        std::vector<dnnl::primitive>& net,
                        std::vector<std::unordered_map<int, dnnl::memory>>& net_args) {
    dnnl::engine cpu_engine;
    dnnl::engine engine_to_use;
    std::unordered_map<dnnl::engine::kind, dnnl::engine>::const_iterator iter = dnnl_engine.find(dnnl::engine::kind::cpu);
    if (iter != dnnl_engine.end()) {
      dnnl_engine_cpu_ = iter->second;
      cpu_engine = iter->second;
      engine_to_use = cpu_engine;
    }
    gpu_available_ = false;
    dnnl::engine gpu_engine;
    iter = dnnl_engine.find(dnnl::engine::kind::gpu);
    if (iter != dnnl_engine.end()) {
      dnnl_engine_gpu_ = iter->second;
      gpu_engine = iter->second;
      gpu_available_ = true;
      engine_to_use = gpu_engine;
      LOGS_DEFAULT(INFO) << "gpu engine found" << std::endl;
    }
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
      x_shape = TensorShape(xshape, xdim);
      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
      ort_source_desc_ = dnnl::memory::desc(
          {dnnl::memory::dims(x_shape.GetDims().begin(), x_shape.GetDims().end())}, DnnnType<T>(), ort_source_format_);
      source_desc_ = ort_source_desc_;
    } else {
      // get the output of previous node (Dnnl block propagation).
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;
    }

    const OrtValue* winput_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    auto wtensor_info = ort.GetTensorTypeAndShape(winput_tensor);
    auto wtensor_shape = ort.GetTensorShape(wtensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(wtensor_info);
    auto wshape = wtensor_shape.data();
    auto wdim = wtensor_shape.size();
    TensorShape w_shape(wshape, wdim);

    weights_shape_ = w_shape;
    weights_format_ = GetSourceFormat(static_cast<int>(w_shape.NumDimensions()));

    std::vector<int64_t> y_dims;
    InferOutputShape(x_shape, w_shape, y_dims);
    primitive_dst_shape_ = TensorShape(y_dims);

    std::unique_ptr<dnnl::memory::desc> src_md = onnxruntime::make_unique<dnnl::memory::desc>(
        dnnl::memory::dims(x_shape.GetDims().begin(), x_shape.GetDims().end()), DnnnType<T>(), dnnl::memory::format_tag::any);

    std::unique_ptr<dnnl::memory::desc> weights_md = onnxruntime::make_unique<dnnl::memory::desc>(
        dnnl::memory::dims(w_shape.GetDims().begin(), w_shape.GetDims().end()), DnnnType<T>(), dnnl::memory::format_tag::any);

    primitive_dst_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
        dnnl::memory::dims(y_dims.begin(), y_dims.end()), DnnnType<T>(), dnnl::memory::format_tag::any);

    std::unique_ptr<dnnl::matmul::desc> matmul_desc = onnxruntime::make_unique<dnnl::matmul::desc>(*src_md, *weights_md, *primitive_dst_md_);
    matmul_pd_ = onnxruntime::make_unique<dnnl::matmul::primitive_desc>(*matmul_desc, engine_to_use);
    matmul_ = onnxruntime::make_unique<dnnl::matmul>(dnnl::matmul(*matmul_pd_));

    primitive_src_desc_ = static_cast<dnnl::memory::desc>(matmul_pd_.get()->src_desc());
    primitive_dst_desc_ = static_cast<dnnl::memory::desc>(matmul_pd_.get()->dst_desc());

    weights_size_ = matmul_pd_.get()->weights_desc().get_size();
    dst_size_ = matmul_pd_.get()->dst_desc().get_size();

    weights_mem_ = onnxruntime::make_unique<dnnl::memory>(
        dnnl::memory(matmul_pd_.get()->weights_desc(), cpu_engine, nullptr));
    if (gpu_available_) {
      weights_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory(matmul_pd_.get()->weights_desc(), gpu_engine, nullptr));
    }

    if (!gpu_available_) {
      if (primitive_src_desc_ != source_desc_) {
        if (mklnode_ptr_->parent_nodes.empty()) {
          dnnl::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
          auto pd = dnnl::memory::desc({{src_dims}, DnnnType<T>(), ort_source_format_});
          src_mem_from_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, nullptr));
        }
        else
          src_mem_from_ = parents_[0].get()->primitive_dst_mem_;

        src_mem_ = onnxruntime::make_unique<dnnl::memory>(
            dnnl::memory(matmul_pd_->src_desc(), cpu_engine, nullptr));
        net.push_back(dnnl::reorder(*src_mem_from_, *src_mem_));
        net_args.push_back({{DNNL_ARG_FROM, *src_mem_from_},
                            {DNNL_ARG_TO, *src_mem_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(matmul_pd_->src_desc(), cpu_engine, nullptr));
        } else {
          src_mem_ = parents_[0].get()->primitive_dst_mem_;
        }
      }

      if (mklnode_ptr_->output_index >= 0) {
        if (primitive_dst_desc_ != ort_source_desc_) {
          primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(matmul_pd_.get()->dst_desc(), cpu_engine));
        } else {
          primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(matmul_pd_.get()->dst_desc(), cpu_engine, nullptr));
        }
      }
    } else {  // gpu_available_
      if (primitive_src_desc_ != source_desc_) {
        if (mklnode_ptr_->parent_nodes.empty()) {
          dnnl::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
          auto pd = dnnl::memory::desc({{src_dims}, DnnnType<T>(), ort_source_format_});
          src_mem_from_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, nullptr));
        } else {
          src_mem_from_ = parents_[0].get()->primitive_dst_mem_;
        }
        src_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(
            dnnl::memory(matmul_pd_->src_desc(), gpu_engine));
        net.push_back(dnnl::reorder(*src_mem_from_, *src_mem_gpu_));
        net_args.push_back({{DNNL_ARG_FROM, *src_mem_from_},
                            {DNNL_ARG_TO, *src_mem_gpu_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(matmul_pd_->src_desc(), cpu_engine, nullptr));
          src_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(matmul_pd_->src_desc(), gpu_engine));
          net.push_back(dnnl::reorder(*src_mem_, *src_mem_gpu_));
          net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                              {DNNL_ARG_DST, *src_mem_gpu_}});
        } else {
          src_mem_gpu_ = parents_[0].get()->primitive_dst_mem_;
        }
      }

      primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory(matmul_pd_.get()->dst_desc(), gpu_engine));
    }

    net.push_back(*matmul_);
    if (!gpu_available_) {
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                          {DNNL_ARG_WEIGHTS, *weights_mem_},
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    } else {  // gpu_available_
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_gpu_},
                          {DNNL_ARG_WEIGHTS, *weights_mem_gpu_},
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    }

    if (mklnode_ptr_->output_index >= 0) {
      dnnl::memory::data_type t = DnnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net, net_args, gpu_available_);
    }
  }

  virtual void ReorderWeights(const OrtCustomOpApi* api, OrtKernelContext* context, const dnnl::engine& cpu_engine) override {
    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
    auto tensor_shape = ort.GetTensorShape(tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

    const T* weights_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));

    dnnl::memory::dims weights_dims_dnnl;
    weights_dims_dnnl.assign(weights_shape_.GetDims().begin(), weights_shape_.GetDims().end());

    {
      // lock to make sure reordering is done only once
      std::lock_guard<OrtMutex> lock(provider_->GetMutex());
      std::shared_ptr<dnnl::memory> weights_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);

      if (weights_dst_mem == nullptr) {
        dnnl::memory src = dnnl::memory({{weights_dims_dnnl}, DnnnType<T>(), weights_format_}, cpu_engine, (void*)weights_data);
        IAllocatorUniquePtr<void> weights_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc_, weights_size_);
        if (!gpu_available_) {
          weights_dst_mem = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(matmul_pd_->weights_desc(), cpu_engine, weights_reorder_buffer.get()));

          dnnl::reorder(src, *weights_dst_mem)
              .execute(cpu_engine, src, *weights_dst_mem);

          provider_->SaveAllocatedMemory(std::move(weights_reorder_buffer));
          weights_data = static_cast<T*>(weights_dst_mem->get_data_handle());
        } else {  // gpu_available_
          weights_dst_mem = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(matmul_pd_->weights_desc(), dnnl_engine_gpu_));

          dnnl::reorder(src, *weights_dst_mem)
              .execute(dnnl_engine_gpu_, src, *weights_dst_mem);
        }

        provider_->SetWeightsMemoryBuffer(mklnode_ptr_->weight_name, weights_dst_mem);
      }
    }
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};

    ORT_RETURN_IF_ERROR(primitive_created_status_);

    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    const OrtValue* winput_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    const T* weights_data = const_cast<T*>(ort.GetTensorData<T>(winput_tensor));

    std::shared_ptr<dnnl::memory> weights_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);
    if (weights_dst_mem == nullptr) {
      ReorderWeights(api, context, dnnl_engine_cpu_);
      weights_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);
    }
    if (!gpu_available_) {
      weights_data = static_cast<T*>(weights_dst_mem->get_data_handle());
      weights_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(weights_data)));
    } else {  // gpu_available_
      weights_mem_gpu_->set_data_handle(weights_dst_mem->get_data_handle());
    }

    if (primitive_src_desc_ != source_desc_) {
      if (mklnode_ptr_->parent_nodes.empty()) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
        const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
        src_mem_from_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
      } else {
        src_mem_from_ = parents_[0].get()->primitive_dst_mem_;
      }

      if (!gpu_available_) {
        auto src_size = matmul_pd_.get()->src_desc().get_size();
        src_reorder_buffer_ = IAllocator::MakeUniquePtr<void>(alloc_, src_size);
        src_mem_->set_data_handle(src_reorder_buffer_.get());
      }
    } else {
      if (mklnode_ptr_->parent_nodes.empty()) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
        const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
        src_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
      } else {
        src_mem_ = parents_[0].get()->primitive_dst_mem_;
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      auto& y_dims = primitive_dst_shape_.GetDims();
      // Allocate memory for output buffer
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
  dnnl::memory::format_tag weights_format_;
  std::shared_ptr<dnnl::memory> src_mem_from_;

  size_t weights_size_;
  size_t dst_size_;
  TensorShape weights_shape_;

  std::shared_ptr<dnnl::memory> src_mem_;
  std::shared_ptr<dnnl::memory> src_mem_gpu_;
  std::shared_ptr<dnnl::memory> weights_mem_;
  std::unique_ptr<dnnl::memory> weights_mem_gpu_;

  std::unique_ptr<dnnl::matmul::primitive_desc> matmul_pd_;
  std::unique_ptr<dnnl::matmul::primitive> matmul_;

  dnnl::engine dnnl_engine_cpu_;
  dnnl::engine dnnl_engine_gpu_;

  bool gpu_available_;

  IAllocatorUniquePtr<void> src_reorder_buffer_;

  void InferOutputShape(const TensorShape& input_shape, const TensorShape& weight_shape, std::vector<int64_t>& output_shape) const {
    output_shape = input_shape.GetDims();
    output_shape.pop_back();
    output_shape.emplace_back(weight_shape.GetDims().back());
    for (size_t i = 0; i < output_shape.size() - 2; i++) {
      if (output_shape[i] == 1) {
        output_shape[i] = weight_shape[i];
      }
    }
  }
};
}  // namespace ort_dnnl
}  // namespace onnxruntime