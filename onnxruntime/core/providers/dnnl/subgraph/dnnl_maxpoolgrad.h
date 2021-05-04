// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"


/*
MaxPoolGrad: (According to OnnxRuntime discovered using code inspection and Onnx documentation)
  Inputs:
    0) dY - Gradient of output Y
    1) indices - indices
  Outputs:
    0) dX - Gradient of Input

                        +-----------------+
    (dY) diff_dst       |                 |
    ------------------->+                 | (dX ) diff_src
    (indices) workspace | MaxPoolGrad     +----------------->
    ------------------->+                 |
                        |                 |
                        +-----------------+

  diff_dst  = DNNL_ARG_DIFF_DST
  workspace = DNNL_ARG_WORKSPACE

  diff_src  = DNNL_ARG_DIFF_SRC

Attributes (auto_pad, dilations, group, kernel_shap, pads, and strides) should be the same as the forward pass Pool operator

The indices must come from the forward pool operator the indices input from OnnxRuntime will be ignored. For that reason the
forward and backward operators must run using dnnl endpoint.
*/
namespace onnxruntime {
namespace ort_dnnl {
template <typename T>
class DnnlMaxPoolGrad : public DnnlKernel {
 public:
  DnnlMaxPoolGrad(const DnnlNode& node,
                  DNNLExecutionProvider* provider,
                  const NodeAttributes& attributes,
                  const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    op_name_ = node.name;
    ReadAttributes(attributes, attributes_prefix);
  }

  void AddForwardDnnlKernel(std::shared_ptr<DnnlPool<T>> pool_fwd) {
    pool_fwd_ = pool_fwd;
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

    Ort::CustomOpApi ort{*api};
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (mklnode_ptr_->parent_nodes.empty()) {
      //First input
      const OrtValue* xgrad_input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto xgrad_tensor_info = ort.GetTensorTypeAndShape(xgrad_input_tensor);
      auto xgrad_tensor_shape = ort.GetTensorShape(xgrad_tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(xgrad_tensor_info);
      auto xgradshape = xgrad_tensor_shape.data();
      auto xgraddim = xgrad_tensor_shape.size();

      dnnl::memory::dims xgraddims(xgraddim);
      xgrad_shape_ = TensorShape(xgradshape, xgraddim);

      dnnl::memory::dims xgrad_src_dims_mkl(xgrad_shape_.GetDims().begin(), xgrad_shape_.GetDims().end());
      ort_source_format_ = GetSourceFormat(static_cast<int>(xgraddim));
      // ort_source_desc is the format of ONNX Runtime tensor format
      ort_source_desc_ = dnnl::memory::desc({xgrad_src_dims_mkl}, DnnnType<T>(), ort_source_format_);
      // source_desc is propagating format. input to this op.
      source_desc_ = dnnl::memory::desc({xgrad_src_dims_mkl}, DnnnType<T>(), ort_source_format_);

      // reorder for better performance
      dnnl::memory::format_tag diff_dst_format = GetAVXFormat(xgrad_src_dims_mkl);
      diff_dst_md_ = std::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc({xgrad_src_dims_mkl}, DnnnType<T>(), diff_dst_format));
    } else {
      // get the output of previous node (Dnnl block propagation).
      // TODO Sourcenode will set src of this node.
      x_shape_ = parents_[0].get()->primitive_dst_shape_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;

      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;
    }

    //Obtain output size and shape from the forward desc in maxpool.
    //This would be the input shape and size in the maxpool forward
    primitive_dst_shape_ = pool_fwd_->GetOutputShape();
    std::vector<int64_t> y_dims = primitive_dst_shape_.GetDims();

    if (xgrad_shape_.NumDimensions() < 3) {
      primitive_created_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                                  "1D Pooling is not supported by DNNL.");
    }

    // TODO Keep this code here for implementing GlobalPool
    /*if (global_pooling_) {
      kernel_shape_.assign(x_dims.begin() + 2, x_dims.end());
      pads_.assign(kernel_shape_.size() * 2, 0);
      strides_.assign(kernel_shape_.size(), 1);
    }*/

    dnnl::memory::dims dst_dims_mkl(y_dims.begin(), y_dims.end());
    dnnl::memory::dims kernel_mkl(kernel_shape_.begin(), kernel_shape_.end());
    dnnl::memory::dims strides_mkl(strides_.begin(), strides_.end());
    dnnl::memory::dims padding_left_mkl(pads_.begin(), pads_.begin() + (pads_.size() / 2));
    dnnl::memory::dims padding_right_mkl(pads_.begin() + (pads_.size() / 2), pads_.end());

    primitive_dst_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({dst_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));

    dnnl::algorithm algo = dnnl::algorithm::pooling_max;
    // TODO Keep this code here for implementing AveragePool
    /*if (op_name_ == "AveragePool" || op_name_ == "GlobalAveragePool") {
      algo = dnnl::algorithm::pooling_avg_exclude_padding;
      if (count_include_pad_) {
        algo = dnnl::algorithm::pooling_avg_include_padding;
      }
    }*/

    bwd_desc_ = std::make_unique<dnnl::pooling_backward::desc>(
        dnnl::pooling_backward::desc(algo, *primitive_dst_md_, *diff_dst_md_,
                                     strides_mkl, kernel_mkl,
                                     padding_left_mkl, padding_right_mkl));
    auto pool_fwd_pd = pool_fwd_->GetPrimitiveDesc();
    bwd_primitive_desc_ = std::make_unique<dnnl::pooling_backward::primitive_desc>(
        dnnl::pooling_backward::primitive_desc(*bwd_desc_, engine_to_use, *pool_fwd_pd));

    primitive_src_desc_ = bwd_primitive_desc_.get()->diff_src_desc();
    primitive_dst_desc_ = bwd_primitive_desc_.get()->diff_dst_desc();

    if (!gpu_available_) {
      // reorder source memory for best performance (AVX512);
      if (primitive_dst_desc_ != source_desc_) {
        dnnl::memory::desc pd(source_desc_);

        if (mklnode_ptr_->parent_nodes.empty())
          diff_dst_mem_from_ = std::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, DNNL_MEMORY_NONE));
        else
          diff_dst_mem_from_ = parents_[0].get()->primitive_dst_mem_;

        diff_dst_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(bwd_primitive_desc_->diff_dst_desc(), cpu_engine, DNNL_MEMORY_NONE));
        net.push_back(dnnl::reorder(*diff_dst_mem_from_, *diff_dst_mem_));
        net_args.push_back({{DNNL_ARG_FROM, *diff_dst_mem_from_},
                            {DNNL_ARG_TO, *diff_dst_mem_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          diff_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(bwd_primitive_desc_->diff_dst_desc(), cpu_engine, DNNL_MEMORY_NONE));
        } else {
          diff_dst_mem_ = parents_[0].get()->primitive_dst_mem_;
        }
      }
    } else {  //gpu_available_
      if (primitive_dst_desc_ != source_desc_) {
        //dnnl::memory::dims src_dims(xgrad_shape_.GetDims().begin(), xgrad_shape_.GetDims().end());
        dnnl::memory::desc pd(source_desc_);

        if (mklnode_ptr_->parent_nodes.empty())
          diff_dst_mem_from_ = std::make_unique<dnnl::memory>(
              dnnl::memory(pd, cpu_engine, DNNL_MEMORY_NONE));
        else
          diff_dst_mem_from_ = parents_[0].get()->primitive_dst_mem_;

        diff_dst_mem_gpu_ = std::make_unique<dnnl::memory>(
            dnnl::memory(bwd_primitive_desc_->diff_dst_desc(), gpu_engine));
        net.push_back(dnnl::reorder(*diff_dst_mem_from_, *diff_dst_mem_gpu_));
        net_args.push_back({{DNNL_ARG_FROM, *diff_dst_mem_from_},
                            {DNNL_ARG_TO, *diff_dst_mem_gpu_}});
      } else {
        if (mklnode_ptr_->parent_nodes.empty()) {
          // Sub-graph's first node. Read input from input buffer
          diff_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(bwd_primitive_desc_->diff_dst_desc(), cpu_engine, DNNL_MEMORY_NONE));
          diff_dst_mem_gpu_ = std::make_unique<dnnl::memory>(
              dnnl::memory(bwd_primitive_desc_->diff_dst_desc(), gpu_engine));
          net.push_back(dnnl::reorder(*diff_dst_mem_, *diff_dst_mem_gpu_));
          net_args.push_back({{DNNL_ARG_SRC, *diff_dst_mem_},
                              {DNNL_ARG_DST, *diff_dst_mem_gpu_}});
        } else {
          diff_dst_mem_gpu_ = parents_[0].get()->primitive_dst_mem_;
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
              dnnl::memory(bwd_primitive_desc_.get()->diff_src_desc(), cpu_engine));
        } else {
          // Last node but re-order not needed. Allocate buffer to output of this node
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(bwd_primitive_desc_.get()->diff_src_desc(), cpu_engine, DNNL_MEMORY_NONE));
        }
      } else {
        // Intermediate node. Use Dnnl kernel internal memory for output and
        // use this as input to next node.
        primitive_dst_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(bwd_primitive_desc_.get()->diff_src_desc(), cpu_engine));
      }
    } else {
      primitive_dst_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(bwd_primitive_desc_.get()->diff_src_desc(), gpu_engine));
    }

    auto workspace_mem_ = pool_fwd_->GetWorkspacePtr();

    pool_bwd_ = std::make_unique<dnnl::pooling_backward>(
        dnnl::pooling_backward(*bwd_primitive_desc_));

    net.push_back(*pool_bwd_);
    if (!gpu_available_) {
      net_args.push_back({{DNNL_ARG_DIFF_DST, *diff_dst_mem_},
                          {DNNL_ARG_DIFF_SRC, *primitive_dst_mem_},
                          {DNNL_ARG_WORKSPACE, *workspace_mem_}});
    } else {
      net_args.push_back({{DNNL_ARG_DIFF_DST, *diff_dst_mem_gpu_},
                          {DNNL_ARG_DIFF_SRC, *primitive_dst_mem_},
                          {DNNL_ARG_WORKSPACE, *workspace_mem_}});
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
    if (bwd_primitive_desc_.get()->diff_dst_desc() != source_desc_) {
      if (mklnode_ptr_->parent_nodes.empty()) {
        const OrtValue* dx_input_tensor = ort.KernelContext_GetInput(context, input_index);
        const T* dx_data = const_cast<T*>(ort.GetTensorData<T>(dx_input_tensor));
        diff_dst_mem_from_->set_data_handle(static_cast<void*>(const_cast<T*>(dx_data)));
      } else {
        diff_dst_mem_from_ = parents_[0].get()->primitive_dst_mem_;
      }
      if (!gpu_available_) {
        auto diff_dst_size = bwd_primitive_desc_.get()->diff_dst_desc().get_size();
        src_reorder_buffer_ = IAllocator::MakeUniquePtr<void>(alloc_, diff_dst_size);
        diff_dst_mem_->set_data_handle(src_reorder_buffer_.get());
      }
    } else {
      if (mklnode_ptr_->parent_nodes.empty()) {
        const OrtValue* dx_input_tensor = ort.KernelContext_GetInput(context, input_index);
        const T* diff_dst_data = const_cast<T*>(ort.GetTensorData<T>(dx_input_tensor));
        diff_dst_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(diff_dst_data)));
      } else {
        diff_dst_mem_ = parents_[0].get()->primitive_dst_mem_;
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      // Last node of sub-graph. Allocate memory for output_buffer data
      // Reorder if needed
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
      } else {
        reorder_dst_mem_to_->set_data_handle(dst_data);
      }
    }
    return Status::OK();
  }

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
  std::shared_ptr<dnnl::memory> diff_dst_mem_;
  std::shared_ptr<dnnl::memory> diff_dst_mem_gpu_;

  std::unique_ptr<dnnl::memory::desc> diff_dst_md_;

  std::unique_ptr<dnnl::pooling_forward::desc> fwd_desc_;

  std::shared_ptr<DnnlPool<T>> pool_fwd_;

  std::unique_ptr<dnnl::pooling_backward::desc> bwd_desc_;
  std::unique_ptr<dnnl::pooling_backward::primitive_desc> bwd_primitive_desc_;
  std::unique_ptr<dnnl::primitive> pool_bwd_;

  std::shared_ptr<dnnl::memory> diff_dst_mem_from_;

  std::unique_ptr<dnnl::memory> dst_mem_from_;
  std::unique_ptr<dnnl::memory> dst_mem_to_;

  bool gpu_available_;
 private:
  dnnl::memory::format_tag GetAVXFormat(const dnnl::memory::dims& src_dims_mkl) {
    bool is_2D = src_dims_mkl.size() == 4;
    bool is_1D = src_dims_mkl.size() == 3;
    dnnl::memory::format_tag fmt = dnnl::memory::format_tag::any;
    if (gpu_available_) {
      if (is_1D)
        fmt = dnnl::memory::format_tag::ncw;
      else
        fmt = is_2D ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::ncdhw;
    } else if (CPUIDInfo::GetCPUIDInfo().HasAVX512f()) {
      if (is_1D)
        fmt = dnnl::memory::format_tag::nCw16c;
      else
        fmt = is_2D ? dnnl::memory::format_tag::nChw16c : dnnl::memory::format_tag::nCdhw16c;
    } else if (CPUIDInfo::GetCPUIDInfo().HasAVX2() && (src_dims_mkl[1] % 8 == 0)) {
      if (is_1D)
        fmt = dnnl::memory::format_tag::nCw8c;
      else
        fmt = is_2D ? dnnl::memory::format_tag::nChw8c : dnnl::memory::format_tag::ncdhw;
    } else {
      if (is_1D)
        fmt = dnnl::memory::format_tag::ncw;
      else
        fmt = is_2D ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::ncdhw;
    }
    return fmt;
  }

  //These functions are no longer used to calculate output size, but are kept in case they are needed in the future.
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
                          &pads->at(kernel_shape_.size() + dim),
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
          static_cast<float>(((in_size - 1) * stride) - (*pad_head + *pad_tail) + kernel));
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
  TensorShape xgrad_shape_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
