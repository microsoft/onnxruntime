// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/providers/mkldnn/mkldnn_fwd.h"
#include "core/providers/cpu/nn/autopad_type.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#include "core/providers/mkldnn/subgraph/mkl_kernel.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace mkl_dnn {
template <typename T>
class MklPool : public MklKernel {
 public:
  MklPool(MklNode& node,
          MKLDNNExecutionProvider* provider,
          std::shared_ptr<MKLContext> mkl_context) : MklKernel(node, provider, mkl_context) {
    op_name_ = node.name;
  }

  Status CreatePrimitives(const ONNXRunTimeTensor* input_tensors,
                          mkldnn::engine& cpu_engine, std::vector<mkldnn::primitive>& net,
                          mkldnn::memory::format& source_format) override {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (mklnode_ptr_->parent_nodes.size() == 0) {
      auto xshape = input_tensors[input_index].shape;
      auto xdim = input_tensors[input_index].ndim;
      mkldnn::memory::dims dims(xdim);
      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
      source_format = ort_source_format_;
      src_format_ = ort_source_format_;
      x_shape_ = TensorShape(xshape, xdim);

      mkldnn::memory::dims src_dims_mkl(x_shape_.GetDims().begin(), x_shape_.GetDims().end());

      // reorder for better performance
      mkldnn::memory::format fmt = GetAVXFormat(src_dims_mkl);
      src_md_.reset(new mkldnn::memory::desc(
          {src_dims_mkl}, MklDnnType<T>(), fmt));
    } else {
      // get the output of previous node (mkldnn block propagation).
      // TODO Sourcenode will set src of this node.
      x_shape_ = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = source_format;
      src_format_ = parents_[0].get()->primitive_dst_format_;
      mkldnn::memory::dims src_dims_mkl(x_shape_.GetDims().begin(), x_shape_.GetDims().end());

      if (src_format_ == ort_source_format_) {
        // reorder for better performance
        mkldnn::memory::format fmt = GetAVXFormat(src_dims_mkl);
        src_md_.reset(new mkldnn::memory::desc(
            {src_dims_mkl}, MklDnnType<T>(), fmt));
      } else {
        src_md_.reset(new mkldnn::memory::desc(
            parents_[0].get()->primitive_dst_mem_.get()->get_primitive_desc().desc()));
      }
    }

    const auto& x_dims = x_shape_.GetDims();
    std::vector<int64_t> y_dims = SetOutputSize(x_shape_, x_shape_[1], &pads_);
    primitive_dst_shape_ = TensorShape(y_dims);

    if (x_shape_.NumDimensions() <= 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Please call default CPU kernel.");
    }

    if (global_pooling_) {
      kernel_shape_.assign(x_dims.begin() + 2, x_dims.end());
      pads_.assign(kernel_shape_.size() * 2, 0);
      strides_.assign(kernel_shape_.size(), 1);
    }

    size_t num_outputs = 1;  //OpKernel::Node().OutputDefs().size(); TODO
    if (num_outputs == 2) {
      ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "can not call cpu default op");
    }

    mkldnn::memory::dims dst_dims_mkl(y_dims.begin(), y_dims.end());
    mkldnn::memory::dims kernel_mkl(kernel_shape_.begin(), kernel_shape_.end());
    mkldnn::memory::dims strides_mkl(strides_.begin(), strides_.end());
    mkldnn::memory::dims padding_left_mkl(pads_.begin(), pads_.begin() + (pads_.size() / 2));
    mkldnn::memory::dims padding_right_mkl(pads_.begin() + (pads_.size() / 2), pads_.end());

    primitive_dst_md_.reset(new mkldnn::memory::desc(
        {dst_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::any));

    mkldnn::algorithm algo = mkldnn::algorithm::pooling_max;
    if (op_name_ == "AveragePool" || op_name_ == "GlobalAveragePool") {
      algo = mkldnn::algorithm::pooling_avg_exclude_padding;
      if (count_include_pad_) {
        algo = mkldnn::algorithm::pooling_avg_include_padding;
      }
    }
    fwd_desc_.reset(new mkldnn::pooling_forward::desc(
        mkldnn::prop_kind::forward_inference, algo,
        *src_md_, *primitive_dst_md_,
        strides_mkl, kernel_mkl,
        padding_left_mkl, padding_right_mkl,
        mkldnn::padding_kind::zero));

    fwd_primitive_desc_.reset(new mkldnn::pooling_forward::primitive_desc(
        *fwd_desc_, cpu_engine));

    if (mklnode_ptr_->parent_nodes.size() == 0) {
      // Sub-graph's first node. Read input from input buffer
      src_mem_.reset(new mkldnn::memory(
          fwd_primitive_desc_.get()->src_primitive_desc(), nullptr));
    } else {
      // Sub-graph's inner node. set input to parent's output
      src_mem_ = parents_[0].get()->primitive_dst_mem_;
    }

    primitive_src_format_ = static_cast<mkldnn::memory::format>(
        fwd_primitive_desc_.get()->src_primitive_desc().desc().data.format);

    primitive_dst_format_ = static_cast<mkldnn::memory::format>(
        fwd_primitive_desc_.get()->dst_primitive_desc().desc().data.format);

    src_size_ = fwd_primitive_desc_.get()->src_primitive_desc().get_size();
    dst_size_ = fwd_primitive_desc_.get()->dst_primitive_desc().get_size();

    // reorder source memory for best performance (AVX512);
    if (primitive_src_format_ != src_format_) {
      mkldnn::memory::dims src_dims_mkl(x_shape_.GetDims().begin(), x_shape_.GetDims().end());
      auto src_md = mkldnn::memory::desc(src_dims_mkl, MklDnnType<T>(), src_format_);
      auto pd = mkldnn::memory::primitive_desc(src_md, cpu_engine);

      if (mklnode_ptr_->parent_nodes.size() == 0)
        src_mem_from_.reset(new mkldnn::memory(pd, nullptr));
      else
        src_mem_from_ = parents_[0].get()->primitive_dst_mem_;

      src_mem_.reset(new mkldnn::memory(fwd_primitive_desc_->src_primitive_desc(), nullptr));
      net.push_back(mkldnn::reorder(*src_mem_from_, *src_mem_));
    } else {
      if (mklnode_ptr_->parent_nodes.size() == 0) {
        src_mem_.reset(new mkldnn::memory(fwd_primitive_desc_->src_primitive_desc(), nullptr));
      } else {
        src_mem_ = parents_[0].get()->primitive_dst_mem_;
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      // last node of sub-graph. need to allocate memory for output_tensor
      if (primitive_dst_format_ != ort_source_format_) {
        // reorder neded. Use primitive output as input to reorder and
        // allocate buffer for reorder output, final output of this subgraph
        primitive_dst_mem_.reset(
            new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc()));
      } else {
        // Last node but re-order not needed. Allocate buffer to output of this node
        primitive_dst_mem_.reset(
            new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc(), nullptr));
      }
    } else {
      // Intermediate node. Use mkldnn kernel internal memory for output and
      // use this as input to next node.
      primitive_dst_mem_.reset(
          new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc()));
    }
    pool_fwd_.reset(
        new mkldnn::pooling_forward(*fwd_primitive_desc_, *src_mem_, *primitive_dst_mem_));

    net.push_back(*pool_fwd_);

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      mkldnn::memory::data_type t = MklDnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net);
    }
    return Status::OK();
  }

  Status Bind(const ONNXRunTimeTensor* input_tensors,
                 ONNXRunTimeTensor* const output_tensors) {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (x_shape_.NumDimensions() <= 3) {
      if (mklnode_ptr_->parent_nodes.size() == 0) {
        // abort as MKLDNN cannot execute this. but
        // ORT try to delete output_tensor buffer data. allocate memory so that it can delete
        // fix for test_averagepool_1d_default node test
        auto xshape = input_tensors[input_index].shape;
        auto xdim = input_tensors[input_index].ndim;
        AllocateOutputTensor(output_tensors, mklnode_ptr_->output_index, xshape, xdim, input_tensors[0].dtype);
      }
      std::cout << "MKLDNN cannot compute shape with dim less than three." << std::endl;
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Please call default CPU kernel.");
    }

    if (primitive_src_format_ != src_format_) {
      if (mklnode_ptr_->parent_nodes.size() == 0)
        src_mem_from_->set_data_handle(input_tensors[input_index].data);
      else
        src_mem_from_ = parents_[0].get()->primitive_dst_mem_;

      auto src_size = fwd_primitive_desc_.get()->src_primitive_desc().get_size();
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
      // Last node of sub-graph. Allocate memory for output_buffer data
      // Reorder if needed
      AllocateMemoryAndReorderIfNeeded(output_tensors, input_tensors[0].dtype);
    }
    return Status::OK();
  }

  void ReadAttributes(const std::unordered_map<std::string,
                                               ONNX_NAMESPACE::AttributeProto>& attributes,
                      const std::string attributes_prefix = "") override {
    global_pooling_ = (op_name_ == "GlobalAveragePool" || op_name_ == "GlobalMaxPool" || op_name_ == "GlobalLpPool");
    global_pooling_ = (op_name_ == "GlobalAveragePool" || op_name_ == "GlobalMaxPool" || op_name_ == "GlobalLpPool");

    if (!global_pooling_) {
      bool attr_read = false;
      auto attr = attributes.find(attributes_prefix + "kernel_shape");
      if (attr != attributes.end()) {
        ONNX_NAMESPACE::AttributeProto proto = attr->second;
        GetIntsAttr(proto, kernel_shape_);
        attr_read = true;
      }
      ORT_ENFORCE(attr_read, "No kernel shape is set.");

      std::string auto_padding;
      attr = attributes.find(attributes_prefix + "auto_pad");
      if (attr != attributes.end() &&
          attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        auto_padding = attr->second.s();
      }
      auto_pad_ = StringToAutoPadType(auto_padding);

      attr_read = false;
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
      attr = attributes.find(attributes_prefix + "strides");
      if (attr != attributes.end()) {
        ONNX_NAMESPACE::AttributeProto proto = attr->second;
        if (GetIntsAttr(proto, strides_) == Status::OK())
          attr_read = true;
      }
      if (!attr_read || strides_.empty()) {
        strides_.resize(kernel_shape_.size(), 1);
      }

      attr = attributes.find(attributes_prefix + "count_include_pad");
      int64_t temp = 0;
      if (attr != attributes.end()) {
        ONNX_NAMESPACE::AttributeProto proto = attr->second;
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
  size_t src_size_;
  size_t dst_size_;

  std::shared_ptr<mkldnn::memory> src_mem_;

  std::unique_ptr<mkldnn::pooling_forward::desc> fwd_desc_;
  std::unique_ptr<mkldnn::memory::desc> src_md_;
  std::unique_ptr<mkldnn::pooling_forward::primitive_desc> fwd_primitive_desc_;
  std::unique_ptr<mkldnn::primitive> pool_fwd_;

  std::shared_ptr<mkldnn::memory> src_mem_from_;
  std::unique_ptr<mkldnn::memory> src_mem_to_;

  std::unique_ptr<mkldnn::memory> dst_mem_from_;
  std::unique_ptr<mkldnn::memory> dst_mem_to_;

 private:
  mkldnn::memory::format GetAVXFormat(const mkldnn::memory::dims& src_dims_mkl) {
    bool is_2D = src_dims_mkl.size() == 4 ? true : false;
    mkldnn::memory::format fmt = mkldnn::memory::format::any;
    if (CPUIDInfo::GetCPUIDInfo().HasAVX512f()) {
      fmt = is_2D ? mkldnn::memory::format::nChw16c : mkldnn::memory::format::nCdhw16c;
    } else if (CPUIDInfo::GetCPUIDInfo().HasAVX2() && (src_dims_mkl[1] % 8 == 0)) {
      fmt = is_2D ? mkldnn::memory::format::nChw8c : mkldnn::memory::format::ncdhw;
    } else {
      fmt = is_2D ? mkldnn::memory::format::nchw : mkldnn::memory::format::ncdhw;
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
}  // namespace mkl_dnn
}  // namespace onnxruntime
