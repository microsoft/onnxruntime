// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {
template <typename T>
class DnnlPool : public DnnlKernel {
 public:
  DnnlPool(const DnnlNode& node,
           DNNLExecutionProvider* provider,
           const NodeAttributes& attributes,
           const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    op_name_ = node.name;
    ReadAttributes(attributes, attributes_prefix);
  }

  void CreatePrimitives(const OrtCustomOpApi* api,
                        OrtKernelContext* context,
                        const std::unordered_map<dnnl::engine::kind, dnnl::engine>& dnnl_engine, std::vector<dnnl::primitive>& net,
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
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      auto xshape = tensor_shape.data();
      auto xdim = tensor_shape.size();

      dnnl::memory::dims dims(xdim);
      x_shape_ = TensorShape(xshape, xdim);

      dnnl::memory::dims src_dims_mkl(x_shape_.GetDims().begin(), x_shape_.GetDims().end());
      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
      // ort_source_desc is the format of ONNX Runtime tensor format
      ort_source_desc_ = dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), ort_source_format_);
      // source_desc is propagating format. input to this op.
      source_desc_ = dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), ort_source_format_);

      // reorder for better performance
      dnnl::memory::format_tag src_format = GetAVXFormat(src_dims_mkl);
      src_md_ = std::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), src_format));
    } else {
      // get the output of previous node (Dnnl block propagation).
      // TODO Sourcenode will set src of this node.
      x_shape_ = parents_[0].get()->primitive_dst_shape_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;

      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;

      if (!gpu_available_) {
        if (source_desc_ == ort_source_desc_) {
          dnnl::memory::dims src_dims_mkl(x_shape_.GetDims().begin(), x_shape_.GetDims().end());
          // reorder for better performance
          dnnl::memory::format_tag fmt = GetAVXFormat(src_dims_mkl);
          src_md_ = std::make_unique<dnnl::memory::desc>(
              dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), fmt));
        } else {
          src_md_ = std::make_unique<dnnl::memory::desc>(
              dnnl::memory::desc(parents_[0].get()->primitive_dst_mem_->get_desc()));
        }
      } else {  // gpu_available_
        src_md_ = std::make_unique<dnnl::memory::desc>(
            dnnl::memory::desc(parents_[0].get()->primitive_dst_mem_->get_desc()));
      }
    }

    const auto& x_dims = x_shape_.GetDims();
    std::vector<int64_t> y_dims = SetOutputSize(x_shape_, x_shape_[1], &pads_);
    primitive_dst_shape_ = TensorShape(y_dims);

#ifndef ENABLE_TRAINING
    if (x_shape_.NumDimensions() <= 3) {
#else
    if (x_shape_.NumDimensions() < 3) {
#endif // ENABLE_TRAINING
      primitive_created_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                                  "1D Pooling is not supported by DNNL.");
    }

    if (global_pooling_) {
      kernel_shape_.assign(x_dims.begin() + 2, x_dims.end());
      pads_.assign(kernel_shape_.size() * 2, 0);
      strides_.assign(kernel_shape_.size(), 1);
    }

    dnnl::memory::dims dst_dims_mkl(y_dims.begin(), y_dims.end());
    dnnl::memory::dims kernel_mkl(kernel_shape_.begin(), kernel_shape_.end());
    dnnl::memory::dims strides_mkl(strides_.begin(), strides_.end());
    dnnl::memory::dims padding_left_mkl(pads_.begin(), pads_.begin() + (pads_.size() / 2));
    dnnl::memory::dims padding_right_mkl(pads_.begin() + (pads_.size() / 2), pads_.end());

    primitive_dst_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({dst_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));

    dnnl::algorithm algo = dnnl::algorithm::pooling_max;
    if (op_name_ == "AveragePool" || op_name_ == "GlobalAveragePool") {
      algo = dnnl::algorithm::pooling_avg_exclude_padding;
      if (count_include_pad_) {
        algo = dnnl::algorithm::pooling_avg_include_padding;
      }
    }

#ifdef ENABLE_TRAINING
    auto prop_kind = dnnl::prop_kind::forward;
#else
    auto prop_kind = dnnl::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

    fwd_desc_ = std::make_unique<dnnl::pooling_forward::desc>(
        dnnl::pooling_forward::desc(prop_kind, algo,
                                    *src_md_, *primitive_dst_md_,
                                    strides_mkl, kernel_mkl,
                                    padding_left_mkl, padding_right_mkl));


    fwd_primitive_desc_ = std::make_unique<dnnl::pooling_forward::primitive_desc>(
        dnnl::pooling_forward::primitive_desc(*fwd_desc_, engine_to_use));

    primitive_src_desc_ = fwd_primitive_desc_.get()->src_desc();
    primitive_dst_desc_ = fwd_primitive_desc_.get()->dst_desc();

    if (!gpu_available_) {
      // reorder source memory for best performance (AVX512);
      if (primitive_src_desc_ != source_desc_) {
        auto pd = dnnl::memory::desc(source_desc_);

        if (mklnode_ptr_->parent_nodes.empty())
          src_mem_from_ = std::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, nullptr));
        else
          src_mem_from_ = parents_[0].get()->primitive_dst_mem_;

        src_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(fwd_primitive_desc_->src_desc(), cpu_engine, nullptr));
        net.push_back(dnnl::reorder(*src_mem_from_, *src_mem_));
        net_args.push_back({{DNNL_ARG_FROM, *src_mem_from_},
                            {DNNL_ARG_TO, *src_mem_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(fwd_primitive_desc_->src_desc(), cpu_engine, nullptr));
        } else {
          src_mem_ = parents_[0].get()->primitive_dst_mem_;
        }
      }
    } else {  // gpu_available_
      if (primitive_src_desc_ != source_desc_) {
        auto pd = dnnl::memory::desc(source_desc_);

        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_from_ = std::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, nullptr));
        } else {
          src_mem_from_ = parents_[0].get()->primitive_dst_mem_;
        }
        src_mem_gpu_ = std::make_unique<dnnl::memory>(
            dnnl::memory(fwd_primitive_desc_->src_desc(), gpu_engine));
        net.push_back(dnnl::reorder(*src_mem_from_, *src_mem_gpu_));
        net_args.push_back({{DNNL_ARG_FROM, *src_mem_from_},
                            {DNNL_ARG_TO, *src_mem_gpu_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          src_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(fwd_primitive_desc_->src_desc(), cpu_engine, nullptr));
          src_mem_gpu_ = std::make_unique<dnnl::memory>(
              dnnl::memory(fwd_primitive_desc_->src_desc(), gpu_engine));
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
        // last node of sub-graph. need to allocate memory for output_tensor
        if (primitive_dst_desc_ != ort_source_desc_) {
          // reorder neded. Use primitive output as input to reorder and
          // allocate buffer for reorder output, final output of this subgraph
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(fwd_primitive_desc_.get()->dst_desc(), cpu_engine));
        } else {
          // Last node but re-order not needed. Allocate buffer to output of this node
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(fwd_primitive_desc_.get()->dst_desc(), cpu_engine, nullptr));
        }
      } else {
        // Intermediate node. Use Dnnl kernel internal memory for output and
        // use this as input to next node.
        primitive_dst_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(fwd_primitive_desc_.get()->dst_desc(), cpu_engine));
      }
    } else {  // gpu_available_
      primitive_dst_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(fwd_primitive_desc_.get()->dst_desc(), gpu_engine));
    }

#ifdef ENABLE_TRAINING
    workspace_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory(fwd_primitive_desc_.get()->workspace_desc(), engine_to_use));
#endif // ENABLE_TRAINING

    pool_fwd_ = std::make_unique<dnnl::pooling_forward>(
        dnnl::pooling_forward(*fwd_primitive_desc_));

    net.push_back(*pool_fwd_);
    if (!gpu_available_) {
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
#ifdef ENABLE_TRAINING
                          {DNNL_ARG_WORKSPACE, *workspace_mem_},
#endif  //ENABLE_TRAINING
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    } else {  // gpu_available_
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_gpu_},
#ifdef ENABLE_TRAINING
                          {DNNL_ARG_WORKSPACE, *workspace_mem_},
#endif  //ENABLE_TRAINING
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    }
    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      dnnl::memory::data_type t = DnnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net, net_args, gpu_available_);
    }
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};

    ORT_RETURN_IF_ERROR(primitive_created_status_);

    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    if (fwd_primitive_desc_.get()->src_desc() != source_desc_) {
      if (mklnode_ptr_->parent_nodes.empty()) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
        const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
        src_mem_from_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
      } else {
        src_mem_from_ = parents_[0].get()->primitive_dst_mem_;
      }

      if (!gpu_available_) {
        auto src_size = fwd_primitive_desc_.get()->src_desc().get_size();
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
      // Last node of sub-graph. Allocate memory for output_buffer data
      // Reorder if needed
      auto& y_dims = primitive_dst_shape_.GetDims();
      // Allocate memory for output bufffer
#ifndef ENABLE_TRAINING
      OrtValue* output = ort.KernelContext_GetOutput(context, mklnode_ptr_->output_index, &y_dims[0], static_cast<int>(primitive_dst_shape_.GetDims().size()));
#else
      OrtValue* output = ort.KernelContext_GetOutput(context, 0, &y_dims[0], static_cast<int>(primitive_dst_shape_.GetDims().size()));
#endif // ENABLE_TRAINING
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
  std::shared_ptr<dnnl::pooling_forward::primitive_desc> GetPrimitiveDesc() {
    return fwd_primitive_desc_;
  }

  std::shared_ptr<dnnl::memory> GetWorkspacePtr() {
    return workspace_mem_;
  }

  TensorShape GetOutputShape() {
    return x_shape_;
  }
  #endif // ENABLE_TRAINING

 private:
  void ReadAttributes(const NodeAttributes& attributes,
                      const std::string attributes_prefix = "") override {
    global_pooling_ = (op_name_ == "GlobalAveragePool" || op_name_ == "GlobalMaxPool" || op_name_ == "GlobalLpPool");
    global_pooling_ = (op_name_ == "GlobalAveragePool" || op_name_ == "GlobalMaxPool" || op_name_ == "GlobalLpPool");

    if (!global_pooling_) {
      bool attr_read = false;
      auto attr = attributes.find(attributes_prefix + "kernel_shape");
      if (attr != attributes.end()) {
        auto& proto = attr->second();
        GetIntsAttr(proto, kernel_shape_);
        attr_read = true;
      }
      ORT_ENFORCE(attr_read, "No kernel shape is set.");

      std::string auto_padding;
      attr = attributes.find(attributes_prefix + "auto_pad");
      if (attr != attributes.end() &&
          attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        auto_padding = attr->second().s();
      }
      auto_pad_ = StringToAutoPadType(auto_padding);

      attr_read = false;
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
      attr = attributes.find(attributes_prefix + "strides");
      if (attr != attributes.end()) {
        auto& proto = attr->second();
        if (GetIntsAttr(proto, strides_) == Status::OK())
          attr_read = true;
      }
      if (!attr_read || strides_.empty()) {
        strides_.resize(kernel_shape_.size(), 1);
      }

      attr = attributes.find(attributes_prefix + "count_include_pad");
      int64_t temp = 0;
      if (attr != attributes.end()) {
        auto& proto = attr->second();
        GetIntAttr(proto, temp);
      }
      count_include_pad_ = (temp != 0);

      storage_order_ = 0;
      for (size_t dim = 0; dim < kernel_shape_.size(); ++dim) {
        ORT_ENFORCE(kernel_shape_[dim] > 0);
        ORT_ENFORCE(pads_[dim] < kernel_shape_[dim] && pads_[dim + kernel_shape_.size()] < kernel_shape_[dim],
                    "Pad should be smaller than kernel.");
      }

      ORT_ENFORCE(strides_.size() == kernel_shape_.size());
    }
  }

 private:

  std::shared_ptr<dnnl::memory> src_mem_;
  std::shared_ptr<dnnl::memory> src_mem_gpu_;
  #ifdef ENABLE_TRAINING
  std::shared_ptr<dnnl::memory> workspace_mem_;
  #endif // ENABLE_TRAINING

  std::unique_ptr<dnnl::pooling_forward::desc> fwd_desc_;
  std::unique_ptr<dnnl::memory::desc> src_md_;
#ifndef ENABLE_TRAINING
  std::unique_ptr<dnnl::pooling_forward::primitive_desc> fwd_primitive_desc_;
#else
  std::unique_ptr<dnnl::memory::desc> workspace_md_;
  std::shared_ptr<dnnl::pooling_forward::primitive_desc> fwd_primitive_desc_;
#endif // ENABLE_TRAINING
  std::unique_ptr<dnnl::primitive> pool_fwd_;

  std::shared_ptr<dnnl::memory> src_mem_from_;
  std::unique_ptr<dnnl::memory> src_mem_to_;

  std::unique_ptr<dnnl::memory> dst_mem_from_;
  std::unique_ptr<dnnl::memory> dst_mem_to_;

  bool gpu_available_;

 private:
  dnnl::memory::format_tag GetAVXFormat(const dnnl::memory::dims& src_dims_mkl) {
    bool is_2D = src_dims_mkl.size() == 4 ? true : false;
#ifdef ENABLE_TRAINING
    bool is_1D = src_dims_mkl.size() == 3 ? true : false;
#endif
    dnnl::memory::format_tag fmt = dnnl::memory::format_tag::any;
    if (gpu_available_) {
#ifdef ENABLE_TRAINING
      if (is_1D)
        fmt = dnnl::memory::format_tag::ncw;
      else
#endif
        fmt = is_2D ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::ncdhw;
    } else if (CPUIDInfo::GetCPUIDInfo().HasAVX512f()) {
#ifdef ENABLE_TRAINING
      if (is_1D)
        fmt = dnnl::memory::format_tag::nCw16c;
      else
#endif
      fmt = is_2D ? dnnl::memory::format_tag::nChw16c : dnnl::memory::format_tag::nCdhw16c;
    } else if (CPUIDInfo::GetCPUIDInfo().HasAVX2() && (src_dims_mkl[1] % 8 == 0)) {
#ifdef ENABLE_TRAINING
      if (is_1D)
        fmt = dnnl::memory::format_tag::nCw8c;
      else
#endif
      fmt = is_2D ? dnnl::memory::format_tag::nChw8c : dnnl::memory::format_tag::ncdhw;
    } else {
#ifdef ENABLE_TRAINING
      if (is_1D)
        fmt = dnnl::memory::format_tag::ncw;
      else
#endif
      fmt = is_2D ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::ncdhw;
    }
    return fmt;
  }

  std::vector<int64_t> SetOutputSize(const TensorShape& input_shape,
                                     int64_t output_channel,
                                     std::vector<int64_t>* pads) const {
    ORT_ENFORCE(input_shape.Size() > 0);
    std::vector<int64_t> output_dims;
    int64_t N = input_shape[0];
    InferOutputSize(input_shape.GetDims(), &output_dims, pads);

    output_dims.insert(output_dims.begin(), {N, output_channel});

    return output_dims;
  }

  inline void InferOutputSize(const std::vector<int64_t>& input_dims,
                              std::vector<int64_t>* output_dims,
                              std::vector<int64_t>* pads) const {
    ORT_ENFORCE(input_dims.size() >= 2);
    if (global_pooling_) {
      output_dims->assign(input_dims.size() - 2, 1);
    } else {
      for (size_t dim = 0; dim < input_dims.size() - 2; ++dim) {
        int64_t dim_size = 0;
        ComputeSizeAndPad(static_cast<int>(input_dims[dim + 2]),
                          strides_[dim],
                          kernel_shape_[dim],
                          &pads->at(dim),
                          &pads->at(input_dims.size() + dim - 2),
                          &dim_size);
        output_dims->push_back(dim_size);
      }
    }
  }

  inline void ComputeSizeAndPad(const int64_t in_size,
                                const int64_t stride,
                                const int64_t kernel,
                                int64_t* pad_head,
                                int64_t* pad_tail,
                                int64_t* out_size) const {
    if (auto_pad_ != AutoPadType::NOTSET) {
      switch (auto_pad_) {
        case AutoPadType::VALID:
          *pad_head = 0;
          *pad_tail = 0;
          *out_size = (in_size - kernel) / stride + 1;
          break;
        case AutoPadType::SAME_LOWER: {
          int64_t legacy_target_size = (in_size + stride - 1) / stride;
          int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
          *pad_head = (pad_needed + 1) / 2;
          *pad_tail = pad_needed - *pad_head;
          *out_size = (in_size + pad_needed - kernel) / stride + 1;
          break;
        }
        case AutoPadType::SAME_UPPER: {
          int64_t legacy_target_size = (in_size + stride - 1) / stride;
          int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
          *pad_head = pad_needed / 2;
          *pad_tail = pad_needed - *pad_head;
          *out_size = (in_size + pad_needed - kernel) / stride + 1;
          break;
        }
        default: {
          ORT_THROW("Unsupported AutoPad Type.");
        }
      }
    } else {
      *out_size = static_cast<int64_t>(
          static_cast<float>(in_size + *pad_head + *pad_tail - kernel) / stride + 1);
    }
  }

 private:
  IAllocatorUniquePtr<void> src_reorder_buffer_;
  IAllocatorUniquePtr<void> dst_reorder_buffer_;

 private:
  std::string op_name_;
  bool global_pooling_{};
  bool count_include_pad_{};
  int64_t storage_order_{0};  // MaxPool_8 only. 0 is row major, and 1 is column major. Default is 0.
  std::vector<int64_t> kernel_shape_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  AutoPadType auto_pad_;

  TensorShape x_shape_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
