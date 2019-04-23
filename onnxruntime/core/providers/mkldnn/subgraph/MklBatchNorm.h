// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"
#include "core/providers/cpu/nn/autopad_type.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#include "core/providers/mkldnn/subgraph/mkl_kernel.h"
#include "core/providers/mkldnn/memcpy_s.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace mkl_dnn {

class BatchNormHelper {
 public:
  static common::Status ValidateInputs(const TensorShape& xshape,
                                       const TensorShape& scale_shape,
                                       const TensorShape& b_shape,
                                       const TensorShape& mean_shape,
                                       const TensorShape& var_shape) {
    // defined as per spec and used for validation
    constexpr int kNumInputScaleDimensions = 1;
    constexpr int kNumInputBiasDimensions = 1;
    constexpr int kNumInputMeanDimensions = 1;
    constexpr int kNumInputVarianceDimensions = 1;

    if (xshape.GetDims().empty()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid input X: Empty dimensions");
    }

    int64_t num_channels = xshape.GetDims()[1];

    if (scale_shape.NumDimensions() != kNumInputScaleDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: NumDimensions() != ", kNumInputScaleDimensions);
    }
    if (scale_shape.GetDims()[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: 0th dimension != ", num_channels);
    }

    if (b_shape.NumDimensions() != kNumInputBiasDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: NumDimensions() != ", kNumInputBiasDimensions);
    }
    if (b_shape.GetDims()[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: 0th dimension != ", num_channels);
    }

    if (mean_shape.NumDimensions() != kNumInputMeanDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: NumDimensions() != ", kNumInputMeanDimensions);
    }
    if (mean_shape.GetDims()[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: 0th dimension != ", num_channels);
    }

    if (var_shape.NumDimensions() != kNumInputVarianceDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: NumDimensions() != ", kNumInputVarianceDimensions);
    }
    if (var_shape.GetDims()[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: 0th dimension != ", num_channels);
    }
    return common::Status::OK();
  }

  static void NormalizeDims(const TensorShape& x_shape, std::vector<int64_t>& new_dims) {
    new_dims.clear();
    auto& orig_dims = x_shape.GetDims();
    if (orig_dims.size() == 4 /*supported size by CUDA*/ ||
        orig_dims.size() == 5 /*supported size by CUDA*/) {
      new_dims = orig_dims;
      return;
    }

    auto rank = x_shape.NumDimensions();
    auto num_samples = rank > 0 ? orig_dims[0] : 1;  // NCHW
    auto num_channels = rank > 1 ? orig_dims[1] : 1;
    auto width = rank > 3 ? orig_dims[3] : 1;
    auto height = rank > 2 ? orig_dims[2] : 1;
    new_dims = {num_samples, num_channels, height, width};
  }
};

template <typename T>
class MklBatchNorm : public MklKernel {
 public:
  explicit MklBatchNorm(MklNode& node,
                        MKLDNNExecutionProvider* provider,
                        std::shared_ptr<MKLContext> mkl_context) : MklKernel(node, provider, mkl_context) {
  }
  void ReadAttributes(const std::unordered_map<std::string,
                                               ONNX_NAMESPACE::AttributeProto>& attributes,
                      const std::string attributes_prefix = "") override {
    auto attr = attributes.find(attributes_prefix + "epsilon");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      epsilon_ = attr->second.f();
    }
  }

  Status CreatePrimitives(const ONNXRunTimeTensor* input_tensors,
                          mkldnn::engine& cpu_engine,
                          std::vector<mkldnn::primitive>& net,
                          mkldnn::memory::format& source_format) override {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.size() == 0) {
      auto xshape = input_tensors[input_index].shape;
      auto xdim = input_tensors[input_index].ndim;
      mkldnn::memory::dims dims(xdim);
      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
      source_format = ort_source_format_;
      src_format_ = ort_source_format_;
      x_shape = TensorShape(xshape, xdim);

      mkldnn::memory::dims src_dims_mkl(
          x_shape.GetDims().begin(), x_shape.GetDims().end());
      src_md_.reset(new mkldnn::memory::desc(
          {src_dims_mkl}, MklDnnType<T>(), src_format_));
    } else {
      src_md_.reset(new mkldnn::memory::desc(parents_[0].get()->primitive_dst_mem_.get()->get_primitive_desc().desc()));
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = source_format;
      src_format_ = parents_[0].get()->primitive_dst_format_;
    }

    int num_dimensions = static_cast<int>(x_shape.NumDimensions());
    if (num_dimensions == 3) {
      primitive_created_ = Status(common::ONNXRUNTIME,
                                  common::NOT_IMPLEMENTED, "BatchNorm: Please call default CPU kernel.");
      return primitive_created_;
    }

    TensorShape scale_shape(input_tensors[input_index + 1].shape, input_tensors[input_index + 1].ndim);
    TensorShape b_shape(input_tensors[input_index + 2].shape, input_tensors[input_index + 2].ndim);
    TensorShape mean_shape(input_tensors[input_index + 3].shape, input_tensors[input_index + 3].ndim);
    TensorShape var_shape(input_tensors[input_index + 4].shape, input_tensors[input_index + 4].ndim);
    primitive_dst_shape_ = TensorShape(x_shape);

    primitive_created_ = BatchNormHelper::ValidateInputs(x_shape, scale_shape, b_shape, mean_shape, var_shape);
    if (!primitive_created_.IsOK())
      return primitive_created_;

    mkldnn::memory::dims src_dims_mkl(
        x_shape.GetDims().begin(), x_shape.GetDims().end());
    mkldnn::memory::dims scale_dims_mkl(
        scale_shape.GetDims().begin(), scale_shape.GetDims().end());
    mkldnn::memory::dims b_dims_mkl(
        b_shape.GetDims().begin(), b_shape.GetDims().end());
    mkldnn::memory::dims mean_dims_mkl(
        mean_shape.GetDims().begin(), mean_shape.GetDims().end());
    mkldnn::memory::dims var_dims_mkl(
        var_shape.GetDims().begin(), var_shape.GetDims().end());

    mkldnn::memory::dims dst_dims_mkl(
        primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());

    scale_shift_md_.reset(new mkldnn::memory::desc(
        {2, scale_dims_mkl[0]}, MklDnnType<T>(), mkldnn::memory::format::nc));
    mean_md_.reset(new mkldnn::memory::desc(
        {mean_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::x));
    var_md_.reset(new mkldnn::memory::desc(
        {var_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::x));
    primitive_dst_md_.reset(new mkldnn::memory::desc(
        {dst_dims_mkl}, MklDnnType<T>(), mkldnn::memory::format::any));

    // scale_shift_mem will allocate 2*C*sizeof(float) buffer
    //
    scale_shift_mem_.reset(
        new mkldnn::memory({*scale_shift_md_, cpu_engine}));

    mean_mem_.reset(
        new mkldnn::memory({*mean_md_, cpu_engine}, nullptr));
    var_mem_.reset(
        new mkldnn::memory({*var_md_, cpu_engine}, nullptr));

    batchnorm_fwd_.reset(new mkldnn::batch_normalization_forward::desc(
        mkldnn::prop_kind::forward_inference, *src_md_, epsilon_,
        mkldnn::batch_normalization_flag::use_scale_shift |
            mkldnn::batch_normalization_flag::use_global_stats));

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

      batchnorm_fwd_pd_.reset(new mkldnn::batch_normalization_forward::primitive_desc(
          *batchnorm_fwd_, attr, cpu_engine));
    } else {
      batchnorm_fwd_pd_.reset(
          new mkldnn::batch_normalization_forward::primitive_desc(
              *batchnorm_fwd_, cpu_engine));
    }

    // out format of this kernel
    primitive_dst_format_ = static_cast<mkldnn::memory::format>(
        batchnorm_fwd_pd_.get()->dst_primitive_desc().desc().data.format);
    primitive_src_format_ = static_cast<mkldnn::memory::format>(
        batchnorm_fwd_pd_.get()->dst_primitive_desc().desc().data.format);

    if (mklnode_ptr_->parent_nodes.size() == 0) {
      src_mem_.reset(
          new mkldnn::memory(batchnorm_fwd_pd_.get()->src_primitive_desc(), nullptr));
    } else {
      src_mem_ = parents_[0].get()->primitive_dst_mem_;
    }

    if (mklnode_ptr_->output_index >= 0) {
      // Use mkldnn's internal output buffer
      if (primitive_dst_format_ != ort_source_format_) {
        primitive_dst_mem_.reset(new mkldnn::memory(batchnorm_fwd_pd_->dst_primitive_desc()));
      } else {
        primitive_dst_mem_.reset(new mkldnn::memory(batchnorm_fwd_pd_->dst_primitive_desc(), nullptr));
      }
    } else {
      // last node of sub-graph. need to allocate memory for output_tensor
      primitive_dst_mem_.reset(new mkldnn::memory(batchnorm_fwd_pd_->dst_primitive_desc()));
    }
    auto bn = mkldnn::batch_normalization_forward(
        *batchnorm_fwd_pd_,
        (const mkldnn::primitive::at)*src_mem_,
        (const mkldnn::primitive::at)*mean_mem_,
        (const mkldnn::primitive::at)*var_mem_,
        (const mkldnn::memory)*scale_shift_mem_,
        (const mkldnn::memory)*primitive_dst_mem_);
    net.push_back(bn);

    // Allocate dst buffer if reorder is necessary
    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      mkldnn::memory::data_type t = MklDnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net);
    }
    return Status::OK();
  }

  Status Submit(const ONNXRunTimeTensor* input_tensors,
                 ONNXRunTimeTensor* const output_tensors) override {
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

    if (mklnode_ptr_->parent_nodes.size() == 0) {
      src_mem_->set_data_handle(input_tensors[input_index].data);
    }

    const T* scale_data = reinterpret_cast<const T*>(input_tensors[input_index + 1].data);
    const T* b_data = reinterpret_cast<const T*>(input_tensors[input_index + 2].data);
    const T* mean_data = reinterpret_cast<const T*>(input_tensors[input_index + 3].data);
    const T* var_data = reinterpret_cast<const T*>(input_tensors[input_index + 4].data);

    TensorShape scale_shape(input_tensors[input_index + 1].shape, input_tensors[input_index + 1].ndim);
    mkldnn::memory::dims scale_dims_mkl(
        scale_shape.GetDims().begin(), scale_shape.GetDims().end());

    mean_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(mean_data)));
    var_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(var_data)));

    T* scale_shift_buf = static_cast<T*>(scale_shift_mem_->get_data_handle());

    size_t src_bytes = sizeof(T) * scale_dims_mkl[0];
    size_t dst_bytes = sizeof(T) * scale_dims_mkl[0];

    MEMCPY_S(scale_shift_buf, scale_data, src_bytes, dst_bytes);
    MEMCPY_S(&scale_shift_buf[scale_dims_mkl[0]], b_data, src_bytes, dst_bytes);

    if (mklnode_ptr_->output_index >= 0) {
      AllocateMemoryAndReorderIfNeeded(output_tensors, input_tensors[0].dtype);
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<mkldnn::memory> src_mem_;
  std::unique_ptr<mkldnn::memory> scale_shift_mem_;
  std::unique_ptr<mkldnn::memory> mean_mem_;
  std::unique_ptr<mkldnn::memory> var_mem_;
  std::unique_ptr<mkldnn::memory> dst_mem_;

  std::unique_ptr<mkldnn::memory::desc> src_md_;
  std::unique_ptr<mkldnn::memory::desc> scale_shift_md_;
  std::unique_ptr<mkldnn::memory::desc> mean_md_;
  std::unique_ptr<mkldnn::memory::desc> var_md_;
  std::unique_ptr<mkldnn::memory::desc> dst_md_;

  std::unique_ptr<mkldnn::batch_normalization_forward::desc> batchnorm_fwd_;
  std::unique_ptr<mkldnn::batch_normalization_forward::primitive_desc> batchnorm_fwd_pd_;

 protected:
  float epsilon_ = 1e-5f;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime