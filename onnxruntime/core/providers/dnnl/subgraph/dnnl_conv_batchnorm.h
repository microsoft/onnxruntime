// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "dnnl_types.h"
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"
#include <cmath>

namespace onnxruntime {
namespace ort_dnnl {

template <typename T>
class DnnlConvBatchNorm : public DnnlKernel {
 public:
  DnnlConvBatchNorm(const DnnlNode& node,
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
      dnnl_engine_cpu_ = (dnnl::engine)iter->second;
      cpu_engine = (dnnl::engine)iter->second;
      engine_to_use = cpu_engine;
    }
    gpu_available_ = false;
    dnnl::engine gpu_engine;
    iter = dnnl_engine.find(dnnl::engine::kind::gpu);
    if (iter != dnnl_engine.end()) {
      dnnl_engine_gpu_ = (dnnl::engine)iter->second;
      gpu_engine = (dnnl::engine)(iter->second);
      gpu_available_ = true;
      engine_to_use = gpu_engine;
      LOGS_DEFAULT(INFO) << "gpu engine found" << std::endl;
    }
    Ort::CustomOpApi ort{*api};
    stream_ = std::make_unique<dnnl::stream>(dnnl::stream(cpu_engine));
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    const OrtValue* winput_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    auto wtensor_info = ort.GetTensorTypeAndShape(winput_tensor);
    auto wtensor_shape = ort.GetTensorShape(wtensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(wtensor_info);
    auto wshape = wtensor_shape.data();
    auto wdim = wtensor_shape.size();

    TensorShape w_shape(wshape, wdim);
    const int group_mkl = static_cast<int>(group_);

    TensorShape x_shape;
    // std::unique_ptr<TensorShape> x_shape;
    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      auto xshape = tensor_shape.data();
      auto xdim = tensor_shape.size();
      ort_source_format_ = dnnl::memory::format_tag::any;
      x_shape = TensorShape(xshape, xdim);
    } else {
      // get the output of previous node (Dnnl block propagation).
      // TODO Sourcenode will set src of this node.
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;
    }

    primitive_created_status_ = ValidateInputShape(x_shape, w_shape);
    if (!primitive_created_status_.IsOK()) {
      return;
    }

    std::vector<int64_t> kernel_shape;
    primitive_created_status_ = ComputeKernelShape(w_shape, kernel_shape);
    if (!primitive_created_status_.IsOK()) {
      return;
    }

    const size_t kernel_rank = kernel_shape.size();

    if (kernel_rank + 2 != wdim) {
      primitive_created_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape num_dims is not compatible with W num_dims.",
                                                  " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                                  " W: ", w_shape.ToString().c_str());
      return;
    }

    for (size_t i = 0; i < kernel_rank; ++i) {
      if (kernel_shape[i] != w_shape[i + 2]) {
        primitive_created_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape is not compatible with W shape.",
                                                    " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                                    " W: ", w_shape.ToString().c_str());
        return;
      }
    }

    std::vector<int64_t> pads(pads_);
    if (pads.empty()) {
      pads.resize(kernel_rank * 2, 0);
    }
    std::vector<int64_t> dilations(dilations_);
    if (dilations.empty()) {
      dilations.resize(kernel_rank, 1);
    }
    std::vector<int64_t> strides(strides_);
    if (strides.empty()) {
      strides.resize(kernel_rank, 1);
    }

    const int64_t N = x_shape[0];
    const int64_t M = w_shape[0];
    std::vector<int64_t> y_dims;
    y_dims.insert(y_dims.begin(), {N, M});
    TensorShape input_shape = x_shape.Slice(2);
    primitive_created_status_ = InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &y_dims);
    if (!primitive_created_status_.IsOK()) {
      return;
    }

    TensorShape y_shape(y_dims);
    primitive_dst_shape_ = TensorShape(y_dims);
    TensorShape output_shape = y_shape.Slice(2);
    dnnl::memory::dims dst_dims_mkl(y_dims.begin(), y_dims.end());
    primitive_dst_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({dst_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));

    dnnl::memory::dims filter_dims_mkl;
    if (group_mkl == 1) {
      filter_dims_mkl.assign(w_shape.GetDims().begin(), w_shape.GetDims().end());
    } else {
      filter_dims_mkl.assign({group_mkl,
                              static_cast<int>(w_shape[0] / group_mkl)});
      filter_dims_mkl.insert(filter_dims_mkl.end(), w_shape.GetDims().begin() + 1, w_shape.GetDims().end());
    }
    dnnl::memory::dims strides_mkl(strides.begin(), strides.end());
    dnnl::memory::dims dilations_mkl(dilations.begin(), dilations.end());
    // Dnnl dilations start from 0 so we need to subtract 1 from each dim.
    for (size_t dim = 0; dim < kernel_rank; dim++) {
      dilations_mkl[dim] -= 1;
    }

    dnnl::memory::dims padding_left_mkl(pads.begin(), pads.begin() + kernel_rank);
    dnnl::memory::dims padding_right_mkl(pads.begin() + kernel_rank, pads.end());
    dnnl::memory::dims bias_dims_mkl;

    int batchNormIndix = (mklnode_ptr_->num_inputs == 7) ? input_index + 3 : input_index + 2;
    if (mklnode_ptr_->num_inputs == 7) {
      const OrtValue* binput_tensor = ort.KernelContext_GetInput(context, input_index + 2);
      auto btensor_info = ort.GetTensorTypeAndShape(binput_tensor);
      auto btensor_shape = ort.GetTensorShape(btensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(btensor_info);
      auto bshape = btensor_shape.data();
      auto bdim = btensor_shape.size();
      TensorShape b_shape(bshape, bdim);
      bias_dims_mkl.assign(b_shape.GetDims().begin(), b_shape.GetDims().end());
    } else {
      const OrtValue* b_input_tensor = ort.KernelContext_GetInput(context, batchNormIndix + 1);
      auto b_tensor_info = ort.GetTensorTypeAndShape(b_input_tensor);
      auto b_tensor_shape = ort.GetTensorShape(b_tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(b_tensor_info);
      auto bshape = b_tensor_shape.data();
      auto bdim = b_tensor_shape.size();
      TensorShape b_shape(bshape, bdim);
      bias_dims_mkl.assign(b_shape.GetDims().begin(), b_shape.GetDims().end());
    }

    auto src_format = dnnl::memory::format_tag::any;
    if (kernel_rank == 1) {
      src_format = dnnl::memory::format_tag::ncw;
      if (group_mkl == 1) {
        filter_format_ = dnnl::memory::format_tag::oiw;
      } else {
        filter_format_ = dnnl::memory::format_tag::goiw;
      }
    } else if (kernel_rank == 2) {
      src_format = dnnl::memory::format_tag::nchw;
      if (group_mkl == 1) {
        filter_format_ = dnnl::memory::format_tag::oihw;
      } else {
        filter_format_ = dnnl::memory::format_tag::goihw;
      }
    } else {
      src_format = dnnl::memory::format_tag::ncdhw;
      if (group_mkl == 1) {
        filter_format_ = dnnl::memory::format_tag::oidhw;
      } else {
        filter_format_ = dnnl::memory::format_tag::goidhw;
      }
    }

    dnnl::memory::dims src_dims_mkl(x_shape.GetDims().begin(), x_shape.GetDims().end());
    if (mklnode_ptr_->parent_nodes.empty()) {
      ort_source_format_ = src_format;
      ort_source_desc_ = dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), src_format);
      source_desc_ = dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), src_format);
    }

    src_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));

    // Set the memory descriptors to format::any to allow DNNL to decide what the optimal memory layout should be
    // for the computation given the input
    filter_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({filter_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));
    bias_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({bias_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));

    dnnl::memory::dims conv_zero_padding = {0, 0};

    fwd_desc_ = std::make_unique<dnnl::convolution_forward::desc>(
        dnnl::convolution_forward::desc(
            dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, *src_md_,
            *filter_md_, *bias_md_, *primitive_dst_md_,
            strides_mkl, dilations_mkl, padding_left_mkl,
            padding_right_mkl));

    if (fuse_relu_) {
      dnnl::primitive_attr attr;
      // attr.set_int_output_round_mode(dnnl::round_mode::round_nearest);
      // Execute RELU as Fuse PostOps
      const float ops_scale = 1.f;
      const float ops_alpha = 0.f;  // relu negative slope
      const float ops_beta = 0.f;
      dnnl::post_ops ops;
      ops.append_eltwise(ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
      attr.set_post_ops(ops);

      conv_fwd_pd_ = std::make_unique<dnnl::convolution_forward::primitive_desc>(
          dnnl::convolution_forward::primitive_desc(*fwd_desc_, attr, engine_to_use));
    } else {
      conv_fwd_pd_ = std::make_unique<dnnl::convolution_forward::primitive_desc>(
          dnnl::convolution_forward::primitive_desc(*fwd_desc_, engine_to_use));
    }

    primitive_src_desc_ = static_cast<dnnl::memory::desc>(
        conv_fwd_pd_.get()->src_desc());

    filter_desc_ = static_cast<dnnl::memory::desc>(
        conv_fwd_pd_.get()->weights_desc());

    primitive_dst_desc_ = static_cast<dnnl::memory::desc>(
        conv_fwd_pd_.get()->dst_desc());

    src_size_ = conv_fwd_pd_.get()->src_desc().get_size();
    filter_size_ = conv_fwd_pd_.get()->weights_desc().get_size();
    dst_size_ = conv_fwd_pd_.get()->dst_desc().get_size();

    filter_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory(conv_fwd_pd_.get()->weights_desc(), cpu_engine, nullptr));
    if (gpu_available_) {
      filter_mem_gpu_ = std::make_unique<dnnl::memory>(
          dnnl::memory(conv_fwd_pd_.get()->weights_desc(), gpu_engine));
    }

    if (!gpu_available_) {
      if (primitive_src_desc_ != source_desc_) {
        dnnl::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
        auto pd = dnnl::memory::desc({{src_dims}, DnnnType<T>(), ort_source_format_});

        if (mklnode_ptr_->parent_nodes.empty())
          src_mem_from_ = std::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, nullptr));
        else
          src_mem_from_ = parents_[0].get()->primitive_dst_mem_;

        src_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(conv_fwd_pd_->src_desc(), cpu_engine, nullptr));
        net.push_back(dnnl::reorder(*src_mem_from_, *src_mem_));
        net_args.push_back({{DNNL_ARG_FROM, *src_mem_from_},
                            {DNNL_ARG_TO, *src_mem_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_->src_desc(), cpu_engine, nullptr));
        } else {
          src_mem_ = parents_[0].get()->primitive_dst_mem_;
        }
      }
    } else {  // gpu_available_
      if (primitive_src_desc_ != source_desc_) {
        dnnl::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
        auto pd = dnnl::memory::desc({{src_dims}, DnnnType<T>(), ort_source_format_});

        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_from_ = std::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, nullptr));
        } else {
          src_mem_from_ = parents_[0].get()->primitive_dst_mem_;
        }
        src_mem_gpu_ = std::make_unique<dnnl::memory>(
            dnnl::memory(conv_fwd_pd_->src_desc(), gpu_engine));
        net.push_back(dnnl::reorder(*src_mem_from_, *src_mem_gpu_));
        net_args.push_back({{DNNL_ARG_FROM, *src_mem_from_},
                            {DNNL_ARG_TO, *src_mem_gpu_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_->src_desc(), cpu_engine, nullptr));
          src_mem_gpu_ = std::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_->src_desc(), gpu_engine));
          net.push_back(dnnl::reorder(*src_mem_, *src_mem_gpu_));
          net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                              {DNNL_ARG_DST, *src_mem_gpu_}});
        } else {
          src_mem_gpu_ = parents_[0].get()->primitive_dst_mem_;
        }
      }
    }

    if (!gpu_available_) {
      if (mklnode_ptr_->output_index >= 0) {
        // Use Dnnl's internal output buffer
        if (primitive_dst_desc_ != ort_source_desc_) {
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_.get()->dst_desc(), cpu_engine));
        } else {
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_.get()->dst_desc(), cpu_engine, nullptr));
        }
      } else {
        // last node of sub-graph. need to allocate memory for output_tensor
        primitive_dst_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(conv_fwd_pd_.get()->dst_desc(), cpu_engine));
      }
    } else {  // gpu_available_
      primitive_dst_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(conv_fwd_pd_.get()->dst_desc(), gpu_engine));
    }

    bias_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory(conv_fwd_pd_.get()->bias_desc(), cpu_engine, nullptr));
    conv_fwd_ = std::make_unique<dnnl::convolution_forward>(
        dnnl::convolution_forward(*conv_fwd_pd_));
    if (!gpu_available_) {
      net.push_back(*conv_fwd_);
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                          {DNNL_ARG_WEIGHTS, *filter_mem_},
                          {DNNL_ARG_BIAS, *bias_mem_},
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    } else {  // gpu_available_
      bias_mem_gpu_ = std::make_unique<dnnl::memory>(
          dnnl::memory(conv_fwd_pd_.get()->bias_desc(), gpu_engine));
      net.push_back(dnnl::reorder(*bias_mem_, *bias_mem_gpu_));
      net_args.push_back({{DNNL_ARG_SRC, *bias_mem_},
                          {DNNL_ARG_DST, *bias_mem_gpu_}});
      net.push_back(*conv_fwd_);
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_gpu_},
                          {DNNL_ARG_WEIGHTS, *filter_mem_gpu_},
                          {DNNL_ARG_BIAS, *bias_mem_gpu_},
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    }

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      dnnl::memory::data_type t = DnnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net, net_args, gpu_available_);
    }
  }

  void GamaInverseVariance(const OrtCustomOpApi* api, OrtKernelContext* context, std::vector<float>& inv_scale_factor, size_t O) {
    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    int batchNormIndix = (mklnode_ptr_->num_inputs == 7) ? input_index + 3 : input_index + 2;
    const OrtValue* scale_input_tensor = ort.KernelContext_GetInput(context, batchNormIndix);
    const T* bn_scale_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(scale_input_tensor));
    const OrtValue* var_input_tensor = ort.KernelContext_GetInput(context, batchNormIndix + 3);
    const T* bn_var_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(var_input_tensor));

    std::vector<float> inv_scale;
    inv_scale.assign(static_cast<size_t>(O), 0.0f);

    float* data = inv_scale_factor.data();
    for (size_t i = 0; i < O; i++) {
      data[i] = bn_scale_data[i] / std::sqrt(bn_var_data[i] + epsilon_);
    }
  }

  void WeightsScaleByAxix(const OrtCustomOpApi* api, OrtKernelContext* context,
                          std::vector<float>& weights_scaled, std::vector<float> inv_scale,
                          TensorShape& W) {
    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    const auto& w_dims = W.GetDims();
    const T* filter_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));

    int batchNormIndix = (mklnode_ptr_->num_inputs == 7) ? input_index + 3 : input_index + 2;
    const OrtValue* scale_input_tensor = ort.KernelContext_GetInput(context, batchNormIndix);
    auto bn_tensor_info = ort.GetTensorTypeAndShape(scale_input_tensor);
    auto bn_tensor_shape = ort.GetTensorShape(bn_tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(bn_tensor_info);
    auto bn_shape = bn_tensor_shape.data();
    auto bn_dim = bn_tensor_shape.size();
    TensorShape bn_scale_shape(bn_shape, bn_dim);
    const auto& bn_dims = bn_scale_shape.GetDims();

    int64_t num = 1;
    int axis = 1;
    for (size_t k = axis; k < w_dims.size(); k++) {
      num *= w_dims[k];
    }

    int64_t w_size = std::accumulate(w_dims.begin(), w_dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    int64_t bn_scale_size = std::accumulate(bn_dims.begin(), bn_dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    int64_t n = w_size / num;

    float* w_scale_data = weights_scaled.data();
    for (auto i = 0; i < n; i++) {
      int index = bn_scale_size == 1 ? 0 : i;
      for (int64_t j = 0; j < num; j++) {
        w_scale_data[i * num + j] = filter_data[i * num + j] * inv_scale[index];
      }
    }
  }

  virtual void ReorderWeights(const OrtCustomOpApi* api, OrtKernelContext* context, const dnnl::engine& cpu_engine) override {
    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
    auto tensor_shape = ort.GetTensorShape(tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
    auto xshape = tensor_shape.data();
    auto xdim = tensor_shape.size();
    TensorShape W(xshape, xdim);

    const int group_mkl = static_cast<int>(group_);
    dnnl::memory::dims filter_dims_mkl;
    if (group_mkl == 1) {
      filter_dims_mkl.assign(W.GetDims().begin(), W.GetDims().end());
    } else {
      filter_dims_mkl.assign({group_mkl,
                              static_cast<int>(W[0] / group_mkl)});
      filter_dims_mkl.insert(filter_dims_mkl.end(), W.GetDims().begin() + 1, W.GetDims().end());
    }

    std::vector<float> inv_scale_factor;
    const auto& w_dims = W.GetDims();
    const size_t O = w_dims[0];
    inv_scale_factor.assign(static_cast<size_t>(O), 0.0f);
    GamaInverseVariance(api, context, inv_scale_factor, O);

    std::vector<float> weights_scaled_by_axis;
    weights_scaled_by_axis.assign(static_cast<size_t>(O), 0.0f);
    auto w_size = std::accumulate(w_dims.begin(), w_dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    weights_scaled_by_axis.assign(static_cast<size_t>(w_size), 0.0f);
    WeightsScaleByAxix(api, context, weights_scaled_by_axis, inv_scale_factor, W);

    {
      // lock to make sure reordering is done only once
      std::lock_guard<OrtMutex> lock(provider_->GetMutex());
      std::shared_ptr<dnnl::memory> filter_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);

      if (filter_dst_mem == nullptr) {
        dnnl::memory src = dnnl::memory({{filter_dims_mkl}, DnnnType<T>(), filter_format_}, cpu_engine, (void*)weights_scaled_by_axis.data());
        IAllocatorUniquePtr<void> filter_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc_, filter_size_);
        if (!gpu_available_) {
          filter_dst_mem = std::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_->weights_desc(), cpu_engine, filter_reorder_buffer.get()));

          dnnl::reorder(src, *filter_dst_mem)
              .execute(cpu_engine, src, *filter_dst_mem);

          provider_->SaveAllocatedMemory(std::move(filter_reorder_buffer));
        } else {  // gpu_available_
          filter_dst_mem = std::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_->weights_desc(), dnnl_engine_gpu_));

          dnnl::reorder(src, *filter_dst_mem)
              .execute(dnnl_engine_gpu_, src, *filter_dst_mem);
        }

        provider_->SetWeightsMemoryBuffer(mklnode_ptr_->weight_name, filter_dst_mem);
      }

      std::shared_ptr<dnnl::memory> bias_mem = provider_->GetBiasMemoryBuffer(mklnode_ptr_->weight_name);
      if (bias_mem == nullptr) {
        auto bias_size = conv_fwd_pd_.get()->bias_desc().get_size();
        IAllocatorUniquePtr<void> bias_buffer = IAllocator::MakeUniquePtr<void>(alloc_, bias_size);
        bias_mem = std::make_unique<dnnl::memory>(
            dnnl::memory(conv_fwd_pd_->bias_desc(), cpu_engine, bias_buffer.get()));
        float* bias_buffer_data = static_cast<float*>(bias_buffer.get());
        if (mklnode_ptr_->num_inputs == 7) {
          const OrtValue* conv_bias_tensor = ort.KernelContext_GetInput(context, input_index + 2);
          const T* conv_bias_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(conv_bias_tensor));

          int batchNormIndix = input_index + 3;
          const OrtValue* b_input_tensor = ort.KernelContext_GetInput(context, batchNormIndix + 1);
          const T* bn_b_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(b_input_tensor));

          const OrtValue* mean_input_tensor = ort.KernelContext_GetInput(context, batchNormIndix + 2);
          const T* bn_mean_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(mean_input_tensor));

          //norm_bias = (conv_bias_arr - mean_arr) * inv_std + bn_b_arr;
          for (size_t j = 0; j < O; j++) {
            bias_buffer_data[j] = (conv_bias_data[j] - bn_mean_data[j]) * inv_scale_factor[j] + bn_b_data[j];
          }
        } else {
          // norm_bias = inv_std * (mean_arr - bn_b_arr);
          int batchNormIndix = (mklnode_ptr_->num_inputs == 7) ? input_index + 3 : input_index + 2;
          const OrtValue* b_input_tensor = ort.KernelContext_GetInput(context, batchNormIndix + 1);
          const T* bn_b_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(b_input_tensor));
          const OrtValue* mean_input_tensor = ort.KernelContext_GetInput(context, batchNormIndix + 2);
          const T* bn_mean_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(mean_input_tensor));

          for (size_t j = 0; j < O; j++) {
            bias_buffer_data[j] = bn_b_data[j] - (bn_mean_data[j] * inv_scale_factor[j]);
          }
        }

        provider_->SaveAllocatedBiasMemory(std::move(bias_buffer));
        provider_->SetBiasMemoryBuffer(mklnode_ptr_->weight_name, bias_mem);
      }
    }
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};

    ORT_RETURN_IF_ERROR(primitive_created_status_);

    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    const OrtValue* winput_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    const T* filter_data = const_cast<T*>(ort.GetTensorData<T>(winput_tensor));

    std::shared_ptr<dnnl::memory> filter_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);
    if (filter_dst_mem == nullptr) {
      ReorderWeights(api, context, dnnl_engine_cpu_);
      filter_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);
    }
    if (!gpu_available_) {
      filter_data = static_cast<T*>(filter_dst_mem->get_data_handle());
      filter_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(filter_data)));
    } else {  // gpu_available_
      filter_mem_gpu_->set_data_handle(filter_dst_mem->get_data_handle());
    }

    std::shared_ptr<dnnl::memory> bias_mem = provider_->GetBiasMemoryBuffer(mklnode_ptr_->weight_name);
    const T* bias_data = static_cast<T*>(bias_mem->get_data_handle());
    bias_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(bias_data)));

    if (primitive_src_desc_ != source_desc_) {
      if (mklnode_ptr_->parent_nodes.empty()) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
        const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
        src_mem_from_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
      } else {
        src_mem_from_ = parents_[0].get()->primitive_dst_mem_;
      }

      if (!gpu_available_) {
        auto src_size = conv_fwd_pd_.get()->src_desc().get_size();
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
  void ReadAttributes(const NodeAttributes& attributes,
                      const std::string attributes_prefix = "") override {
    std::string auto_pad;
    auto attr = attributes.find(attributes_prefix + "auto_pad");
    if (attr != attributes.end() &&
        attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
      auto_pad = attr->second().s();
    }
    auto_pad_ = (auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET;

    kernel_shape_specified_ = false;
    attr = attributes.find(attributes_prefix + "kernel_shape");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      Status status = GetIntsAttr(proto, kernel_shape_);
      kernel_shape_specified_ = true;
    }

    attr = attributes.find(attributes_prefix + "strides");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      Status status = GetIntsAttr(proto, strides_);
    }

    bool attr_read = false;
    attr = attributes.find(attributes_prefix + "pads");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      if (GetIntsAttr(proto, pads_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      pads_.resize(kernel_shape_.size() * 2, 0);
    }

    attr_read = false;
    attr = attributes.find(attributes_prefix + "dilations");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      if (GetIntsAttr(proto, dilations_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      dilations_.resize(kernel_shape_.size(), 1);
    }

    attr_read = false;
    attr = attributes.find(attributes_prefix + "group");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      if (GetIntAttr(proto, group_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      group_ = 1;
    }

    attr = attributes.find(attributes_prefix + "epsilon");
    if (attr != attributes.end()) {
      epsilon_ = attr->second().f();
    }
  }

 private:
  dnnl::memory::desc filter_desc_;
  dnnl::memory::format_tag filter_format_;

  std::shared_ptr<dnnl::memory> src_mem_from_;
  std::unique_ptr<dnnl::memory> src_mem_to_;

  size_t src_size_;
  size_t filter_size_;
  size_t dst_size_;

  std::shared_ptr<dnnl::memory> src_mem_;
  std::unique_ptr<dnnl::memory> filter_mem_;
  std::unique_ptr<dnnl::memory> bias_mem_;

  std::shared_ptr<dnnl::memory> src_mem_gpu_;
  std::unique_ptr<dnnl::memory> filter_mem_gpu_;
  std::unique_ptr<dnnl::memory> bias_mem_gpu_;

  std::unique_ptr<dnnl::convolution_forward::desc> fwd_desc_;

  std::unique_ptr<dnnl::memory::desc> src_md_;
  std::unique_ptr<dnnl::memory::desc> filter_md_;
  std::unique_ptr<dnnl::memory::desc> bias_md_;

  std::unique_ptr<dnnl::convolution_forward::primitive_desc> conv_fwd_pd_;
  std::unique_ptr<dnnl::primitive> conv_fwd_;

  dnnl::engine dnnl_engine_cpu_;
  dnnl::engine dnnl_engine_gpu_;

  bool gpu_available_;

 private:
  IAllocatorUniquePtr<void> src_reorder_buffer_;
  IAllocatorUniquePtr<void> dst_reorder_buffer_;

 private:
  Status ComputeKernelShape(const TensorShape& weight_shape, std::vector<int64_t>& kernel_shape) const {
    if (kernel_shape_specified_) {
      kernel_shape = kernel_shape_;
      if (kernel_shape.size() + 2 != weight_shape.NumDimensions()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape num_dims is not compatible with W num_dims.",
                               " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                               " W: ", weight_shape.ToString().c_str());
      }
      for (size_t i = 0; i < kernel_shape.size(); ++i) {
        if (kernel_shape[i] != weight_shape[i + 2]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape is not compatible with W shape.",
                                 " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                 " W: ", weight_shape.ToString().c_str());
        }
      }
    } else {
      auto& weight_dims = weight_shape.GetDims();
      kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    }

    return Status::OK();
  }

  Status ValidateInputShape(const TensorShape& X, const TensorShape& W) const {
    const int64_t C = X[1];
    const int64_t M = W[0];

    if (X.NumDimensions() != W.NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "X num_dims does not match W num_dims.",
                             " X: ", X.ToString().c_str(),
                             " W: ", W.ToString().c_str());
    }

    if (C != W[1] * group_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input channels C is not equal to kernel channels * group.",
                             " C: ", C,
                             " kernel channels: ", W[1],
                             " group: ", group_);
    }

    if (M % group_ != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output channels M is not divisible by group.",
                             " M: ", M,
                             " group: ", group_);
    }
    return Status::OK();
  }

  template <bool ForceSymmetricAutoPadding = false>
  Status InferOutputShape(const TensorShape& input_shape,
                          const std::vector<int64_t>& kernel_shape,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& dilations,
                          std::vector<int64_t>* pads,
                          std::vector<int64_t>* output_shape) const {
    size_t rank = gsl::narrow_cast<int>(input_shape.NumDimensions());
    for (size_t dim = 0; dim < rank; ++dim) {
      if (dim >= strides.size() || dim >= kernel_shape.size() ||
          dim >= dilations.size() || dim >= pads->size() ||
          rank + dim >= pads->size()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Out of bound access to array");
      }
      int64_t dim_size = 0;
      ORT_RETURN_IF_ERROR(ComputePadAndOutputShape<ForceSymmetricAutoPadding>(
          input_shape[dim],
          strides[dim],
          kernel_shape[dim],
          dilations[dim],
          auto_pad_,
          &pads->at(dim),
          &pads->at(input_shape.NumDimensions() + dim),
          &dim_size));
      if (dim_size <= 0) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid input shape: " + input_shape.ToString());
      }
      output_shape->push_back(dim_size);
    }
    return Status::OK();
  }

 private:
  std::unique_ptr<dnnl::stream> stream_;
  std::vector<int64_t> kernel_shape_;  // must use ComputeKernelShape(...), instead of kernel_shape_
  AutoPadType auto_pad_;
  int64_t group_;
  bool kernel_shape_specified_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> dilations_;
  std::string activation_;
  float alpha_;
  float epsilon_ = 1e-5f;  // attribute of fused batchnorm.
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
