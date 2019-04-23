// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "mkldnn_types.h"
#include "core/framework/op_kernel.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"
#include "core/providers/cpu/nn/autopad_type.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#include "core/providers/mkldnn/subgraph/mkl_kernel.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace mkl_dnn {

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
class MklConv : public MklKernel {
 public:
  MklConv(MklNode& node,
          MKLDNNExecutionProvider* provider,
          std::shared_ptr<MKLContext> mkl_context) : MklKernel(node, provider, mkl_context) {
  }

  void ReadAttributes(const std::unordered_map<std::string,
                                               ONNX_NAMESPACE::AttributeProto>& attributes,
                      const std::string attributes_prefix = "") override {
    std::string auto_pad;
    auto attr = attributes.find(attributes_prefix + "auto_pad");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
      auto_pad = attr->second.s();
    }
    auto_pad_ = (auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET;

    kernel_shape_specified_ = false;
    attr = attributes.find(attributes_prefix + "kernel_shape");
    if (attr != attributes.end()) {
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      Status status = GetIntsAttr(proto, kernel_shape_);
      kernel_shape_specified_ = true;
    }

    attr = attributes.find(attributes_prefix + "strides");
    if (attr != attributes.end()) {
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      Status status = GetIntsAttr(proto, strides_);
    }

    bool attr_read = false;
    attr = attributes.find(attributes_prefix + "pads");
    if (attr != attributes.end()) {
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      if (GetIntsAttr(proto, pads_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      pads_.resize(kernel_shape_.size() * 2, 0);
    }

    attr_read = false;
    attr = attributes.find(attributes_prefix + "dilations");
    if (attr != attributes.end()) {
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      if (GetIntsAttr(proto, dilations_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      dilations_.resize(kernel_shape_.size(), 1);
    }

    attr_read = false;
    attr = attributes.find(attributes_prefix + "group");
    if (attr != attributes.end()) {
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      if (GetIntAttr(proto, group_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      group_ = 1;
    }
  }

  Status CreatePrimitives(const ONNXRunTimeTensor* input_tensors,
                          mkldnn::engine& cpu_engine,
                          std::vector<mkldnn::primitive>& net,
                          mkldnn::memory::format& source_format) override {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    TensorShape w_shape(input_tensors[input_index + 1].shape, input_tensors[input_index + 1].ndim);
    const int group_mkl = static_cast<int>(group_);

    TensorShape x_shape;
    // std::unique_ptr<TensorShape> x_shape;
    if (mklnode_ptr_->parent_nodes.size() == 0) {
      auto xshape = input_tensors[input_index].shape;
      auto xdim = input_tensors[input_index].ndim;
      mkldnn::memory::dims dims(xdim);
      x_shape = TensorShape(xshape, xdim);

      mkldnn::memory::dims src_dims_mkl(x_shape.GetDims().begin(), x_shape.GetDims().end());
      src_md_.reset(new mkldnn::memory::desc(
          {src_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::any));
    } else {
      // get the output of previous node (mkldnn block propagation).
      // TODO Sourcenode will set src of this node.
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = source_format;
      src_format_ = parents_[0].get()->primitive_dst_format_;
      mkldnn::memory::format fmt = mkldnn::memory::format::any;
      if (parents_[0].get()->primitive_dst_format_ != ort_source_format_)
        fmt = parents_[0].get()->primitive_dst_format_;
      mkldnn::memory::dims src_dims_mkl(x_shape.GetDims().begin(), x_shape.GetDims().end());
      src_md_.reset(new mkldnn::memory::desc(
          {src_dims_mkl}, MklDnnType<T>(), fmt));
    }

    primitive_created_ = ValidateInputShape(x_shape, w_shape);
    if (!primitive_created_.IsOK())
      return primitive_created_;

    std::vector<int64_t> kernel_shape;
    primitive_created_ = ComputeKernelShape(w_shape, kernel_shape);
    if (!primitive_created_.IsOK())
      return primitive_created_;

    const size_t kernel_rank = kernel_shape.size();

    if (kernel_rank + 2 != input_tensors[input_index + 1].ndim) {
      primitive_created_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape num_dims is not compatible with W num_dims.",
                             " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                             " W: ", w_shape.ToString().c_str());
      return primitive_created_;
    }

    for (size_t i = 0; i < kernel_rank; ++i) {
      if (kernel_shape[i] != w_shape[i + 2]) {
        primitive_created_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape is not compatible with W shape.",
                               " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                               " W: ", w_shape.ToString().c_str());
        return primitive_created_;
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
    primitive_created_ = InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &y_dims);
    if (!primitive_created_.IsOK())
      return primitive_created_;

    TensorShape y_shape(y_dims);
    primitive_dst_shape_ = TensorShape(y_dims);
    TensorShape output_shape = y_shape.Slice(2);
    mkldnn::memory::dims dst_dims_mkl(y_dims.begin(), y_dims.end());
    primitive_dst_md_.reset(new mkldnn::memory::desc(
        {dst_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::any));

    mkldnn::memory::dims filter_dims_mkl;
    if (group_mkl == 1) {
      filter_dims_mkl.assign(w_shape.GetDims().begin(), w_shape.GetDims().end());
    } else {
      filter_dims_mkl.assign({group_mkl,
                              static_cast<int>(w_shape[0] / group_mkl)});
      filter_dims_mkl.insert(filter_dims_mkl.end(), w_shape.GetDims().begin() + 1, w_shape.GetDims().end());
    }
    mkldnn::memory::dims strides_mkl(strides.begin(), strides.end());
    mkldnn::memory::dims dilations_mkl(dilations.begin(), dilations.end());
    // mkldnn dilations start from 0 so we need to subtract 1 from each dim.
    for (size_t dim = 0; dim < kernel_rank; dim++) {
      dilations_mkl[dim] -= 1;
    }

    mkldnn::memory::dims padding_left_mkl(pads.begin(), pads.begin() + kernel_rank);
    mkldnn::memory::dims padding_right_mkl(pads.begin() + kernel_rank, pads.end());
    mkldnn::memory::dims bias_dims_mkl;
    if (mklnode_ptr_->num_inputs == 3) {
      auto bshape = input_tensors[input_index + 2].shape;
      auto bdim = input_tensors[input_index + 2].ndim;
      TensorShape b_shape(bshape, bdim);
      bias_dims_mkl.assign(b_shape.GetDims().begin(), b_shape.GetDims().end());
    }

    auto fmt = mkldnn::memory::format::any;
    if (kernel_rank == 1) {
      fmt = mkldnn::memory::format::ncw;
      if (group_mkl == 1) {
        filter_format_ = mkldnn::memory::format::oiw;
      } else {
        filter_format_ = mkldnn::memory::format::goiw;
      }
    } else if (kernel_rank == 2) {
      fmt = mkldnn::memory::format::nchw;
      if (group_mkl == 1) {
        filter_format_ = mkldnn::memory::format::oihw;
      } else {
        filter_format_ = mkldnn::memory::format::goihw;
      }
    } else {
      fmt = mkldnn::memory::format::ncdhw;
      if (group_mkl == 1) {
        filter_format_ = mkldnn::memory::format::oidhw;
      } else {
        filter_format_ = mkldnn::memory::format::goidhw;
      }
    }
    if (src_format_ == mkldnn::memory::format::any) {
      src_format_ = fmt;
      ort_source_format_ = fmt;
      source_format = fmt;
    }

    // Set the memory descriptors to format::any to allow MKLDNN to decide what the optimal memory layout should be
    // for the computation given the input
    filter_md_.reset(new mkldnn::memory::desc(
        {filter_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::any));
    if (!bias_dims_mkl.empty())
      bias_md_.reset(new mkldnn::memory::desc(
          {bias_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::any));

    if (!bias_dims_mkl.empty()) {
      fwd_desc_.reset(new mkldnn::convolution_forward::desc(
          mkldnn::prop_kind::forward, mkldnn::convolution_direct, *src_md_,
          *filter_md_, *bias_md_, *primitive_dst_md_,
          strides_mkl, dilations_mkl, padding_left_mkl,
          padding_right_mkl, mkldnn::padding_kind::zero));
    } else {
      fwd_desc_.reset(new mkldnn::convolution_forward::desc(
          mkldnn::prop_kind::forward, mkldnn::convolution_direct, *src_md_,
          *filter_md_, *primitive_dst_md_, strides_mkl,
          dilations_mkl, padding_left_mkl,
          padding_right_mkl, mkldnn::padding_kind::zero));
    }

    if (fuse_relu_) {
      mkldnn::primitive_attr attr;
      attr.set_int_output_round_mode(mkldnn::round_mode::round_nearest);
      // Execute RELU as Fuse PostOps
      const float ops_scale = 1.f;
      const float ops_alpha = 0.f;  // relu negative slope
      const float ops_beta = 0.f;
      mkldnn::post_ops ops;
      ops.append_eltwise(ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
      attr.set_post_ops(ops);

      conv_fwd_pd_.reset(new mkldnn::convolution_forward::primitive_desc(
          *fwd_desc_, attr, cpu_engine));
    } else {
      conv_fwd_pd_.reset(new mkldnn::convolution_forward::primitive_desc(
          *fwd_desc_, cpu_engine));
    }

    primitive_src_format_ = static_cast<mkldnn::memory::format>(
        conv_fwd_pd_.get()->src_primitive_desc().desc().data.format);

    mkldnn_filter_format_ = static_cast<mkldnn::memory::format>(
        conv_fwd_pd_.get()->weights_primitive_desc().desc().data.format);

    primitive_dst_format_ = static_cast<mkldnn::memory::format>(
        conv_fwd_pd_.get()->dst_primitive_desc().desc().data.format);

    src_size_ = conv_fwd_pd_.get()->src_primitive_desc().get_size();
    filter_size_ = conv_fwd_pd_.get()->weights_primitive_desc().get_size();
    dst_size_ = conv_fwd_pd_.get()->dst_primitive_desc().get_size();

    filter_mem_.reset(
        new mkldnn::memory(conv_fwd_pd_.get()->weights_primitive_desc(), nullptr));

    if (primitive_src_format_ != src_format_) {
      mkldnn::memory::dims src_dims_mkl(x_shape.GetDims().begin(), x_shape.GetDims().end());
      auto src_md = mkldnn::memory::desc(src_dims_mkl, MklDnnType<T>(), src_format_);
      auto pd = mkldnn::memory::primitive_desc(src_md, cpu_engine);

      if (mklnode_ptr_->parent_nodes.size() == 0)
        src_mem_from_.reset(new mkldnn::memory(pd, nullptr));
      else
        src_mem_from_ = parents_[0].get()->primitive_dst_mem_;

      src_mem_.reset(new mkldnn::memory(conv_fwd_pd_->src_primitive_desc(), nullptr));
      net.push_back(mkldnn::reorder(*src_mem_from_, *src_mem_));
    } else {
      if (mklnode_ptr_->parent_nodes.size() == 0) {
        src_mem_.reset(new mkldnn::memory(conv_fwd_pd_->src_primitive_desc(), nullptr));
      } else {
        src_mem_ = parents_[0].get()->primitive_dst_mem_;
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      // Use mkldnn's internal output buffer
      if (primitive_dst_format_ != ort_source_format_) {
        primitive_dst_mem_.reset(new mkldnn::memory(conv_fwd_pd_.get()->dst_primitive_desc()));
      } else {
        primitive_dst_mem_.reset(new mkldnn::memory(conv_fwd_pd_.get()->dst_primitive_desc(), nullptr));
      }
    } else {
      // last node of sub-graph. need to allocate memory for output_tensor
      primitive_dst_mem_.reset(new mkldnn::memory(conv_fwd_pd_.get()->dst_primitive_desc()));
    }

    if (!bias_dims_mkl.empty()) {
      bias_mem_.reset(new mkldnn::memory(conv_fwd_pd_.get()->bias_primitive_desc(), nullptr));
      conv_fwd_.reset(new mkldnn::convolution_forward(*conv_fwd_pd_, *src_mem_, *filter_mem_,
                                                      *bias_mem_, *primitive_dst_mem_));
    } else {
      conv_fwd_.reset(new mkldnn::convolution_forward(*conv_fwd_pd_, *src_mem_,
                                                      *filter_mem_, *primitive_dst_mem_));
    }
    net.push_back(*conv_fwd_);

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      mkldnn::memory::data_type t = MklDnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net);
    }
    primitive_created_ = Status::OK();
    return primitive_created_;
  }

  virtual void ReorderWeights(const ONNXRunTimeTensor* input_tensors, mkldnn::engine& cpu_engine) override {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    auto xshape = input_tensors[input_index + 1].shape;
    auto xdim = input_tensors[input_index + 1].ndim;
    TensorShape W(xshape, xdim);
    const T* filter_data = reinterpret_cast<const T*>(input_tensors[input_index + 1].data);

    const int group_mkl = static_cast<int>(group_);

    mkldnn::memory::dims filter_dims_mkl;
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
      std::shared_ptr<mkldnn::memory> filter_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);

      if (filter_dst_mem == nullptr) {
        auto pd = mkldnn::memory::primitive_desc(
            mkldnn::memory::desc(filter_dims_mkl, MklDnnType<T>(), filter_format_), cpu_engine);
        mkldnn::memory src = mkldnn::memory(pd, (void*)filter_data);
        IAllocatorUniquePtr<void> filter_reorder_buffer =
            IAllocator::MakeUniquePtr<void>(alloc_, filter_size_);
        filter_dst_mem.reset(
            new mkldnn::memory(conv_fwd_pd_->weights_primitive_desc(), filter_reorder_buffer.get()));

        MemoryReorderParams params(src, *filter_dst_mem);
        DoReorder<T>(params);
        provider_->SaveAllocatedMemory(std::move(filter_reorder_buffer));
        filter_data = static_cast<T*>(filter_dst_mem->get_data_handle());
        provider_->SetWeightsMemoryBuffer(mklnode_ptr_->weight_name, filter_dst_mem);
      }
    }
  }

  Status Submit(const ONNXRunTimeTensor* input_tensors,
                 ONNXRunTimeTensor* const output_tensors) {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;
    if (!primitive_created_.IsOK()) {
      // abort as MKLDNN cannot execute this. but
      // ORT try to delete output_tensor buffer data. allocate memory so that it can delete
      // fix for test_averagepool_1d_default node test
      auto xshape = input_tensors[input_index].shape;
      auto xdim = input_tensors[input_index].ndim;
      AllocateOutputTensor(output_tensors, mklnode_ptr_->output_index, xshape, xdim, input_tensors[0].dtype);
      return primitive_created_;
	}

    const T* filter_data = reinterpret_cast<const T*>(input_tensors[input_index + 1].data);
    const T* bias_data = mklnode_ptr_->num_inputs == 3 ? reinterpret_cast<const T*>(input_tensors[input_index + 2].data) : nullptr;

    std::shared_ptr<mkldnn::memory> filter_dst_mem = provider_->GetWeightsMemoryBuffer(mklnode_ptr_->weight_name);
    filter_data = static_cast<T*>(filter_dst_mem->get_data_handle());

    filter_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(filter_data)));
    if (bias_data != nullptr) {
      bias_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(bias_data)));
    }

    if (primitive_src_format_ != src_format_) {
      if (mklnode_ptr_->parent_nodes.size() == 0)
        src_mem_from_->set_data_handle(input_tensors[input_index].data);
      else
        src_mem_from_ = parents_[0].get()->primitive_dst_mem_;

      auto src_size = conv_fwd_pd_.get()->src_primitive_desc().get_size();
      src_reorder_buffer_ = IAllocator::MakeUniquePtr<void>(alloc_, src_size);
      src_mem_->set_data_handle(src_reorder_buffer_.get());
    } else {
      if (mklnode_ptr_->parent_nodes.size() == 0) {
        src_mem_->set_data_handle(input_tensors[input_index].data);
      } else {
        src_mem_ = parents_[0].get()->primitive_dst_mem_;
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      AllocateMemoryAndReorderIfNeeded(output_tensors, input_tensors[0].dtype);
    }
    return Status::OK();
  }

 private:
  mkldnn::memory::format mkldnn_filter_format_;
  mkldnn::memory::format filter_format_;

  std::shared_ptr<mkldnn::memory> src_mem_from_;
  std::unique_ptr<mkldnn::memory> src_mem_to_;

  size_t src_size_;
  size_t filter_size_;
  size_t dst_size_;

  std::shared_ptr<mkldnn::memory> src_mem_;
  std::unique_ptr<mkldnn::memory> filter_mem_;
  std::unique_ptr<mkldnn::memory> bias_mem_;

  std::unique_ptr<mkldnn::convolution_forward::desc> fwd_desc_;

  std::unique_ptr<mkldnn::memory::desc> src_md_;
  std::unique_ptr<mkldnn::memory::desc> filter_md_;
  std::unique_ptr<mkldnn::memory::desc> bias_md_;

  std::unique_ptr<mkldnn::convolution_forward::primitive_desc> conv_fwd_pd_;
  std::unique_ptr<mkldnn::primitive> conv_fwd_;

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
    int rank = gsl::narrow_cast<int>(input_shape.NumDimensions());
    for (int dim = 0; dim < rank; ++dim) {
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
}  // namespace mkl_dnn
}  // namespace onnxruntime
