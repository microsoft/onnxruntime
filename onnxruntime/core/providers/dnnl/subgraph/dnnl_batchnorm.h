// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"
#include "core/providers/dnnl/memcpy_s.h"

namespace onnxruntime {
namespace ort_dnnl {

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
class DnnlBatchNorm : public DnnlKernel {
 public:
  explicit DnnlBatchNorm(const DnnlNode& node,
                         DNNLExecutionProvider* provider,
                         const NodeAttributes& attributes,
                         const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ReadAttributes(attributes, attributes_prefix);
  }
  void ReadAttributes(const NodeAttributes& attributes,
                      const std::string attributes_prefix = "") override {
    auto attr = attributes.find(attributes_prefix + "epsilon");
    if (attr != attributes.end() &&
        attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      epsilon_ = attr->second().f();
    }
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
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      auto xshape = tensor_shape.data();
      auto xdim = tensor_shape.size();
      dnnl::memory::dims dims(xdim);
      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));

      x_shape = TensorShape(xshape, xdim);

      dnnl::memory::dims src_dims(
          x_shape.GetDims().begin(), x_shape.GetDims().end());

      ort_source_desc_ = dnnl::memory::desc(
          {src_dims}, DnnnType<T>(), ort_source_format_);
      source_desc_ = ort_source_desc_;
      src_md_ = std::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc({src_dims}, DnnnType<T>(), ort_source_format_));
    } else {
      src_md_ = std::make_unique<dnnl::memory::desc>(
          dnnl::memory::desc(parents_[0].get()->primitive_dst_desc_));
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;
    }

    int num_dimensions = static_cast<int>(x_shape.NumDimensions());
    if (num_dimensions == 3) {
      primitive_created_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                                  "1D BatchNormalization is not supported in DNNL.");
      return;
    }

    const OrtValue* scale_input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    const OrtValue* b_input_tensor = ort.KernelContext_GetInput(context, input_index + 2);
    const OrtValue* mean_input_tensor = ort.KernelContext_GetInput(context, input_index + 3);
    const OrtValue* var_input_tensor = ort.KernelContext_GetInput(context, input_index + 4);

    auto scale_tensor_info = ort.GetTensorTypeAndShape(scale_input_tensor);
    auto scale_tensor_shape = ort.GetTensorShape(scale_tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
    auto sshape = scale_tensor_shape.data();
    auto sdim = scale_tensor_shape.size();
    TensorShape scale_shape(sshape, sdim);

    auto b_tensor_info = ort.GetTensorTypeAndShape(b_input_tensor);
    auto b_tensor_shape = ort.GetTensorShape(b_tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(b_tensor_info);
    auto bshape = b_tensor_shape.data();
    auto bdim = b_tensor_shape.size();
    TensorShape b_shape(bshape, bdim);

    auto mean_tensor_info = ort.GetTensorTypeAndShape(mean_input_tensor);
    auto mean_tensor_shape = ort.GetTensorShape(mean_tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(mean_tensor_info);
    auto mshape = mean_tensor_shape.data();
    auto mdim = mean_tensor_shape.size();
    TensorShape mean_shape(mshape, mdim);

    auto var_tensor_info = ort.GetTensorTypeAndShape(var_input_tensor);
    auto var_tensor_shape = ort.GetTensorShape(var_tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(var_tensor_info);
    auto vshape = var_tensor_shape.data();
    auto vdim = var_tensor_shape.size();
    TensorShape var_shape(vshape, vdim);

    primitive_dst_shape_ = TensorShape(x_shape);

    primitive_created_status_ = BatchNormHelper::ValidateInputs(x_shape, scale_shape, b_shape, mean_shape, var_shape);
    if (!primitive_created_status_.IsOK()) {
      return;
    }

    dnnl::memory::dims src_dims_mkl(
        x_shape.GetDims().begin(), x_shape.GetDims().end());
    dnnl::memory::dims scale_dims_mkl(
        scale_shape.GetDims().begin(), scale_shape.GetDims().end());
    dnnl::memory::dims b_dims_mkl(
        b_shape.GetDims().begin(), b_shape.GetDims().end());
    dnnl::memory::dims mean_dims_mkl(
        mean_shape.GetDims().begin(), mean_shape.GetDims().end());
    dnnl::memory::dims var_dims_mkl(
        var_shape.GetDims().begin(), var_shape.GetDims().end());

    dnnl::memory::dims dst_dims_mkl(
        primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());

    scale_shift_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({2, scale_dims_mkl[0]}, DnnnType<T>(), dnnl::memory::format_tag::nc));
    mean_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({mean_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::x));
    var_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({var_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::x));
    primitive_dst_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({dst_dims_mkl}, DnnnType<T>(), dnnl::memory::format_tag::any));

    // scale_shift_mem will allocate 2*C*sizeof(float) buffer
    //
    scale_shift_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory({*scale_shift_md_, cpu_engine}));

    mean_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory(*mean_md_, cpu_engine, nullptr));
    var_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory(*var_md_, cpu_engine, nullptr));

    if (gpu_available_) {
      scale_shift_mem_gpu_ = std::make_unique<dnnl::memory>(
          dnnl::memory({*scale_shift_md_, gpu_engine}));
      mean_mem_gpu_ = std::make_unique<dnnl::memory>(
          dnnl::memory(*mean_md_, gpu_engine));
      var_mem_gpu_ = std::make_unique<dnnl::memory>(
          dnnl::memory(*var_md_, gpu_engine));
    }

    batchnorm_fwd_ = std::make_unique<dnnl::batch_normalization_forward::desc>(
        dnnl::batch_normalization_forward::desc(
            dnnl::prop_kind::forward_inference, *src_md_, epsilon_,
            dnnl::normalization_flags::use_scale_shift |
                dnnl::normalization_flags::use_global_stats));

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

      batchnorm_fwd_pd_ = std::make_unique<dnnl::batch_normalization_forward::primitive_desc>(
          dnnl::batch_normalization_forward::primitive_desc(*batchnorm_fwd_, attr, engine_to_use));
    } else {
      batchnorm_fwd_pd_ = std::make_unique<dnnl::batch_normalization_forward::primitive_desc>(
          dnnl::batch_normalization_forward::primitive_desc(
              *batchnorm_fwd_, engine_to_use));
    }

    // out format of this kernel
    primitive_dst_desc_ = static_cast<dnnl::memory::desc>(
        batchnorm_fwd_pd_.get()->dst_desc());
    primitive_src_desc_ = static_cast<dnnl::memory::desc>(
        batchnorm_fwd_pd_.get()->dst_desc());

    if (!gpu_available_) {
      if (mklnode_ptr_->parent_nodes.empty()) {
        src_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(batchnorm_fwd_pd_.get()->src_desc(), cpu_engine, nullptr));
      } else {
        src_mem_ = parents_[0].get()->primitive_dst_mem_;
      }
    } else {  // gpu_available_
      if (mklnode_ptr_->parent_nodes.empty()) {
        src_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(batchnorm_fwd_pd_.get()->src_desc(), cpu_engine, nullptr));
        src_mem_gpu_ = std::make_unique<dnnl::memory>(
            dnnl::memory(batchnorm_fwd_pd_.get()->src_desc(), gpu_engine));
        net.push_back(dnnl::reorder(*src_mem_, *src_mem_gpu_));
        net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                            {DNNL_ARG_DST, *src_mem_gpu_}});
      } else {
        src_mem_gpu_ = parents_[0].get()->primitive_dst_mem_;
      }
    }

    if (!gpu_available_) {
      if (mklnode_ptr_->output_index >= 0) {
        // Use Dnnl's internal output buffer
        if (primitive_dst_desc_ != ort_source_desc_) {
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(batchnorm_fwd_pd_->dst_desc(), cpu_engine));
        } else {
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(batchnorm_fwd_pd_->dst_desc(), cpu_engine, nullptr));
        }
      } else {
        // last node of sub-graph. need to allocate memory for output_tensor
        primitive_dst_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(batchnorm_fwd_pd_->dst_desc(), cpu_engine));
      }
    } else {  // gpu_available_
      primitive_dst_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(batchnorm_fwd_pd_->dst_desc(), gpu_engine));
    }

    if (gpu_available_) {
      net.push_back(dnnl::reorder(*mean_mem_, *mean_mem_gpu_));
      net_args.push_back({{DNNL_ARG_SRC, *mean_mem_},
                          {DNNL_ARG_DST, *mean_mem_gpu_}});
      net.push_back(dnnl::reorder(*var_mem_, *var_mem_gpu_));
      net_args.push_back({{DNNL_ARG_SRC, *var_mem_},
                          {DNNL_ARG_DST, *var_mem_gpu_}});
      net.push_back(dnnl::reorder(*scale_shift_mem_, *scale_shift_mem_gpu_));
      net_args.push_back({{DNNL_ARG_SRC, *scale_shift_mem_},
                          {DNNL_ARG_DST, *scale_shift_mem_gpu_}});
    }

    auto bn = dnnl::batch_normalization_forward(
        *batchnorm_fwd_pd_);
    net.push_back(bn);
    if (!gpu_available_) {
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                          {DNNL_ARG_MEAN, *mean_mem_},
                          {DNNL_ARG_VARIANCE, *var_mem_},
                          {DNNL_ARG_SCALE_SHIFT, *scale_shift_mem_},
                          {DNNL_ARG_DST, *primitive_dst_mem_}});
    } else {  // gpu_available_
      net_args.push_back({{DNNL_ARG_SRC, *src_mem_gpu_},
                          {DNNL_ARG_MEAN, *mean_mem_gpu_},
                          {DNNL_ARG_VARIANCE, *var_mem_gpu_},
                          {DNNL_ARG_SCALE_SHIFT, *scale_shift_mem_gpu_},
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
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    // abort as DNNL cannot execute this. but
    // ORT try to delete output_tensor buffer data. allocate memory so that it can delete
    // fix for test_averagepool_1d_default node test
    ORT_RETURN_IF_ERROR(primitive_created_status_);

    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
      src_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
    }

    const OrtValue* scale_input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    const T* scale_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(scale_input_tensor));
    const OrtValue* b_input_tensor = ort.KernelContext_GetInput(context, input_index + 2);
    const T* b_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(b_input_tensor));
    const OrtValue* mean_input_tensor = ort.KernelContext_GetInput(context, input_index + 3);
    const T* mean_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(mean_input_tensor));
    const OrtValue* var_input_tensor = ort.KernelContext_GetInput(context, input_index + 4);
    const T* var_data = reinterpret_cast<const T*>(ort.GetTensorData<T>(var_input_tensor));

    auto tensor_info = ort.GetTensorTypeAndShape(scale_input_tensor);
    auto tensor_shape = ort.GetTensorShape(tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
    auto sshape = tensor_shape.data();
    auto sdim = tensor_shape.size();

    TensorShape scale_shape(sshape, sdim);
    dnnl::memory::dims scale_dims_mkl(
        scale_shape.GetDims().begin(), scale_shape.GetDims().end());

    mean_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(mean_data)));
    var_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(var_data)));

    T* scale_shift_buf = static_cast<T*>(scale_shift_mem_->get_data_handle());

    size_t src_bytes = sizeof(T) * scale_dims_mkl[0];
    size_t dst_bytes = sizeof(T) * scale_dims_mkl[0];

    MEMCPY_S(scale_shift_buf, scale_data, src_bytes, dst_bytes);
    MEMCPY_S(&scale_shift_buf[scale_dims_mkl[0]], b_data, src_bytes, dst_bytes);

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
  std::shared_ptr<dnnl::memory> src_mem_;
  std::unique_ptr<dnnl::memory> scale_shift_mem_;
  std::unique_ptr<dnnl::memory> mean_mem_;
  std::unique_ptr<dnnl::memory> var_mem_;
  std::unique_ptr<dnnl::memory> dst_mem_;

  std::shared_ptr<dnnl::memory> src_mem_gpu_;
  std::unique_ptr<dnnl::memory> scale_shift_mem_gpu_;
  std::unique_ptr<dnnl::memory> mean_mem_gpu_;
  std::unique_ptr<dnnl::memory> var_mem_gpu_;

  std::unique_ptr<dnnl::memory::desc> src_md_;
  std::unique_ptr<dnnl::memory::desc> scale_shift_md_;
  std::unique_ptr<dnnl::memory::desc> mean_md_;
  std::unique_ptr<dnnl::memory::desc> var_md_;
  std::unique_ptr<dnnl::memory::desc> dst_md_;

  std::unique_ptr<dnnl::batch_normalization_forward::desc> batchnorm_fwd_;
  std::unique_ptr<dnnl::batch_normalization_forward::primitive_desc> batchnorm_fwd_pd_;

  bool gpu_available_;

 protected:
  float epsilon_ = 1e-5f;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime
