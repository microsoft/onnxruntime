// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "mkldnn_types.h"
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {

// helper function
template <bool ForceSymmetricAutoPadding>
Status ComputePadAndOutputShape(
    const int64_t in_dim,
    const int64_t stride,
    const int64_t kernel,
    const int64_t dilation,
    AutoPadType pad_type,
    int64_t* pad_head,
    int64_t* pad_tail,
    int64_t* out_dim) {
  const int64_t dkernel = dilation * (kernel - 1) + 1;

  if (pad_type == AutoPadType::NOTSET) {
    *out_dim = static_cast<int64_t>(static_cast<float>(in_dim + *pad_head + *pad_tail - dkernel) / stride + 1);
  } else {
    switch (pad_type) {
      case AutoPadType::VALID:
        *pad_head = 0;
        *pad_tail = 0;
        *out_dim = (in_dim - dkernel) / stride + 1;
        break;
      case AutoPadType::SAME_UPPER:
      case AutoPadType::SAME_LOWER: {
        ORT_ENFORCE(dilation == 1, "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.");
        int64_t legacy_target_size = (in_dim + stride - 1) / stride;
        int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
        *out_dim = (in_dim + pad_needed - dkernel) / stride + 1;

        // make sure padding is symmetric
        if (ForceSymmetricAutoPadding)
          pad_needed = math::roundUpPow2<int64_t, 2>(pad_needed);

        if (pad_type == AutoPadType::SAME_LOWER) {
          *pad_head = (pad_needed + 1) / 2;
        } else {
          *pad_head = pad_needed / 2;
        }
        *pad_tail = pad_needed - *pad_head;
      } break;
      default:
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "pad type not supported.");
    }
  }
  return Status::OK();
}

template <typename T>
class DnnlConv : public DnnlKernel {
 public:
  DnnlConv(const DnnlNode& node,
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
    stream_ = onnxruntime::make_unique<dnnl::stream>(dnnl::stream(cpu_engine));

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

      dnnl::memory::dims src_dims_mkl(x_shape.GetDims().begin(), x_shape.GetDims().end());
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
    primitive_dst_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
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
    if (mklnode_ptr_->num_inputs == 3) {
      const OrtValue* binput_tensor = ort.KernelContext_GetInput(context, input_index + 2);
      auto btensor_info = ort.GetTensorTypeAndShape(binput_tensor);
      auto btensor_shape = ort.GetTensorShape(btensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(btensor_info);
      auto bshape = btensor_shape.data();
      auto bdim = btensor_shape.size();
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

    src_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));

    // Set the memory descriptors to format::any to allow DNNL to decide what the optimal memory layout should be
    // for the computation given the input
    filter_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({filter_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));
    if (!bias_dims_mkl.empty())
      bias_md_ = onnxruntime::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc({bias_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));

    dnnl::memory::dims conv_zero_padding = {0, 0};

#ifdef ENABLE_TRAINING
    auto prop_kind = dnnl::prop_kind::forward_training;
#else
    auto prop_kind = dnnl::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

    if (!bias_dims_mkl.empty()) {
      fwd_desc_ = onnxruntime::make_unique<dnnl::convolution_forward::desc>(
          dnnl::convolution_forward::desc(
              prop_kind, dnnl::algorithm::convolution_direct, *src_md_,
              *filter_md_, *bias_md_, *primitive_dst_md_,
              strides_mkl, dilations_mkl, padding_left_mkl,
              padding_right_mkl));
    } else {
      fwd_desc_ = onnxruntime::make_unique<dnnl::convolution_forward::desc>(
          dnnl::convolution_forward::desc(
              prop_kind, dnnl::algorithm::convolution_direct, *src_md_,
              *filter_md_, *primitive_dst_md_, strides_mkl,
              dilations_mkl, padding_left_mkl, padding_right_mkl));
    }

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

      conv_fwd_pd_ = onnxruntime::make_unique<dnnl::convolution_forward::primitive_desc>(
          dnnl::convolution_forward::primitive_desc(*fwd_desc_, attr, engine_to_use));
    } else {
      conv_fwd_pd_ = onnxruntime::make_unique<dnnl::convolution_forward::primitive_desc>(
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

    filter_mem_ = onnxruntime::make_unique<dnnl::memory>(
        dnnl::memory(conv_fwd_pd_.get()->weights_desc(), cpu_engine, nullptr));
    if (gpu_available_) {
      filter_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory(conv_fwd_pd_.get()->weights_desc(), gpu_engine, nullptr));
    }

    if (!gpu_available_) {
      if (primitive_src_desc_ != source_desc_) {
        dnnl::memory::dims src_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
        auto pd = dnnl::memory::desc({{src_dims}, DnnnType<T>(), ort_source_format_});

        if (mklnode_ptr_->parent_nodes.empty())
          src_mem_from_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, nullptr));
        else
          src_mem_from_ = parents_[0].get()->primitive_dst_mem_;

        src_mem_ = onnxruntime::make_unique<dnnl::memory>(
            dnnl::memory(conv_fwd_pd_->src_desc(), cpu_engine, nullptr));
        net.push_back(dnnl::reorder(*src_mem_from_, *src_mem_));
        net_args.push_back({{DNNL_ARG_FROM, *src_mem_from_},
                            {DNNL_ARG_TO, *src_mem_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_ = onnxruntime::make_unique<dnnl::memory>(
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
          src_mem_from_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, nullptr));
        } else {
          src_mem_from_ = parents_[0].get()->primitive_dst_mem_;
        }
        src_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(
            dnnl::memory(conv_fwd_pd_->src_desc(), gpu_engine));
        net.push_back(dnnl::reorder(*src_mem_from_, *src_mem_gpu_));
        net_args.push_back({{DNNL_ARG_FROM, *src_mem_from_},
                            {DNNL_ARG_TO, *src_mem_gpu_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_->src_desc(), cpu_engine, nullptr));
          src_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(
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
          primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_.get()->dst_desc(), cpu_engine));
        } else {
          primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_.get()->dst_desc(), cpu_engine, nullptr));
        }
      } else {
        // last node of sub-graph. need to allocate memory for output_tensor
        primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
            dnnl::memory(conv_fwd_pd_.get()->dst_desc(), cpu_engine));
      }
    } else {  // gpu_available_
      primitive_dst_mem_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory(conv_fwd_pd_.get()->dst_desc(), gpu_engine));
    }

    if (!bias_dims_mkl.empty()) {
      bias_mem_ = onnxruntime::make_unique<dnnl::memory>(
          dnnl::memory(conv_fwd_pd_.get()->bias_desc(), cpu_engine, nullptr));
      conv_fwd_ = onnxruntime::make_unique<dnnl::convolution_forward>(
          dnnl::convolution_forward(*conv_fwd_pd_));
      if (!gpu_available_) {
        net.push_back(*conv_fwd_);
        net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                            {DNNL_ARG_WEIGHTS, *filter_mem_},
                            {DNNL_ARG_BIAS, *bias_mem_},
                            {DNNL_ARG_DST, *primitive_dst_mem_}});
      } else {  // gpu_available_
        bias_mem_gpu_ = onnxruntime::make_unique<dnnl::memory>(
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
    } else {
      conv_fwd_ = onnxruntime::make_unique<dnnl::convolution_forward>(
          dnnl::convolution_forward(*conv_fwd_pd_));
      if (!gpu_available_) {
        net.push_back(*conv_fwd_);
        net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                            {DNNL_ARG_WEIGHTS, *filter_mem_},
                            {DNNL_ARG_DST, *primitive_dst_mem_}});
      } else {  // gpu_available_
        net.push_back(*conv_fwd_);
        net_args.push_back({{DNNL_ARG_SRC, *src_mem_gpu_},
                            {DNNL_ARG_WEIGHTS, *filter_mem_gpu_},
                            {DNNL_ARG_DST, *primitive_dst_mem_}});
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
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
    auto xshape = tensor_shape.data();
    auto xdim = tensor_shape.size();

    TensorShape W(xshape, xdim);
    const T* filter_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));

    const int group_mkl = static_cast<int>(group_);

    dnnl::memory::dims filter_dims_mkl;
    if (group_mkl == 1) {
      filter_dims_mkl.assign(W.GetDims().begin(), W.GetDims().end());
    } else {
      filter_dims_mkl.assign({group_mkl,
                              static_cast<int>(W[0] / group_mkl)});
      filter_dims_mkl.insert(filter_dims_mkl.end(), W.GetDims().begin() + 1, W.GetDims().end());
    }

    {
      // lock to make sure reordering is done only once
      std::lock_guard<OrtMutex> lock(provider_->GetMutex());
      std::shared_ptr<dnnl::memory> filter_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);

      if (filter_dst_mem == nullptr) {
        dnnl::memory src = dnnl::memory({{filter_dims_mkl}, DnnnType<T>(), filter_format_}, cpu_engine, (void*)filter_data);
        IAllocatorUniquePtr<void> filter_reorder_buffer = IAllocator::MakeUniquePtr<void>(alloc_, filter_size_);
        if (!gpu_available_) {
          filter_dst_mem = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_->weights_desc(), cpu_engine, filter_reorder_buffer.get()));

          dnnl::reorder(src, *filter_dst_mem)
              .execute(cpu_engine, src, *filter_dst_mem);

          provider_->SaveAllocatedMemory(std::move(filter_reorder_buffer));
          filter_data = static_cast<T*>(filter_dst_mem->get_data_handle());
        } else {  // gpu_available_
          filter_dst_mem = onnxruntime::make_unique<dnnl::memory>(
              dnnl::memory(conv_fwd_pd_->weights_desc(), dnnl_engine_gpu_));

          dnnl::reorder(src, *filter_dst_mem)
              .execute(dnnl_engine_gpu_, src, *filter_dst_mem);
        }

        // Do not use cached weights if running training since weight is changed each iteration
#ifndef ENABLE_TRAINING
        provider_->SetWeightsMemoryBuffer(mklnode_ptr_->weight_name, filter_dst_mem);
#else
        filter_dst_mem_ = filter_dst_mem;
#endif  // !ENABLE_TRAINING
      }
    }
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};

    ORT_RETURN_IF_ERROR(primitive_created_status_);

    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    const OrtValue* winput_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    const T* filter_data = const_cast<T*>(ort.GetTensorData<T>(winput_tensor));

    const T* bias_data = nullptr;
    if (mklnode_ptr_->num_inputs == 3) {
      const OrtValue* binput_tensor = ort.KernelContext_GetInput(context, input_index + 2);
      bias_data = const_cast<T*>(ort.GetTensorData<T>(binput_tensor));
    }
    // Do not use cached weights if running training
#ifndef ENABLE_TRAINING
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
#else  // ENABLE_TRAINING
    if (!gpu_available_) {
      filter_data = static_cast<T*>(filter_dst_mem_->get_data_handle());
      filter_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(filter_data)));
    } else if (gpu_available_) {
      filter_mem_gpu_->set_data_handle(filter_dst_mem_->get_data_handle());
    }
#endif  // ENABLE_TRAINING

    if (bias_data != nullptr) {
      bias_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(bias_data)));
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
#ifdef ENABLE_TRAINING
  std::shared_ptr<dnnl::convolution_forward::primitive_desc> GetPrimitiveDesc() {
    return conv_fwd_pd_;
  }
#endif  // ENABLE_TRAINING

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
  }

 private:
  dnnl::memory::desc filter_desc_;
  dnnl::memory::format_tag filter_format_;
#ifdef ENABLE_TRAINING
  std::shared_ptr<dnnl::memory> filter_dst_mem_;
#endif  // ENABLE_TRAINING

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

#ifndef ENABLE_TRAINING
  std::unique_ptr<dnnl::convolution_forward::primitive_desc> conv_fwd_pd_;
#else
  std::shared_ptr<dnnl::convolution_forward::primitive_desc> conv_fwd_pd_;
#endif  // ENABLE_TRAINING

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
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
