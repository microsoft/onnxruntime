// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class FusedConv : public onnxruntime::cuda::Conv<T> {
 public:
  FusedConv(const OpKernelInfo& info) : onnxruntime::cuda::Conv<T>(info) {
    std::string activation;
    if (info.GetAttr<std::string>("activation", &activation) == Status::OK() &&
        MapMode(activation) == Status::OK() &&
        cudnnCreateActivationDescriptor(&activation_desc_) == CUDNN_STATUS_SUCCESS) {
      status_ = cudnnSetActivationDescriptor(activation_desc_,
                                             activation_mode_,
                                             cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                                             std::numeric_limits<double>::max());
    }
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FusedConv);

  ~FusedConv() {
    if (activation_desc_) {
      cudnnDestroyActivationDescriptor(activation_desc_);
      status_ = CUDNN_STATUS_NOT_INITIALIZED;
      activation_desc_ = nullptr;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    CUDNN_RETURN_IF_ERROR(status_);
    std::lock_guard<OrtMutex> lock(s_.mutex);
    ORT_RETURN_IF_ERROR(UpdateState(context, true));
    if (s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    bool has_z = nullptr != s_.z_data;
    bool has_b = nullptr != s_.b_data;
    IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(s_.workspace_bytes);
    auto cudnn_status = cudnnConvolutionBiasActivationForward(CudnnHandle(),
                                                              &alpha_,
                                                              s_.x_tensor,
                                                              s_.x_data,
                                                              s_.w_desc,
                                                              s_.w_data,
                                                              s_.conv_desc,
                                                              s_.algo,
                                                              workspace.get(),
                                                              s_.workspace_bytes,
                                                              has_z ? &alpha_ : &beta_,
                                                              has_z ? s_.z_tensor : s_.y_tensor,
                                                              has_z ? s_.z_data : s_.y_data,
                                                              s_.b_tensor,
                                                              has_b ? s_.b_data : s_.b_zero,
                                                              activation_desc_,
                                                              s_.y_tensor,
                                                              s_.y_data);
    if (CUDNN_STATUS_BAD_PARAM == cudnn_status) {
      std::cout << "x: ";
      s_.x_tensor.print();
      std::cout << "b: ";
      s_.b_tensor.print();
      std::cout << "z: ";
      s_.z_tensor.print();
      std::cout << "y: ";
      s_.y_tensor.print();
      std::cout << "has_z: " << (has_z ? "true" : "false") << std::endl;
      std::cout << "has_b: " << (has_b ? "true" : "false") << std::endl;
    }
    CUDNN_RETURN_IF_ERROR(cudnn_status);
    if (s_.post_slicing_required) {
      onnxruntime::cuda::SliceOutUnwantedOutputSection(s_.y_data, s_.y_dims_with_adjusted_pads, s_.Y->MutableDataRaw(),
                                                       s_.y_dims, s_.slice_starts, s_.slice_ends, s_.slice_axes, s_.element_size);
    }
    return Status::OK();
  }

 private:
  Status MapMode(const std::string& activaton_mode) {
    if (activaton_mode == "Relu") {
      activation_mode_ = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU;
    } else {
      return Status(common::StatusCategory::ONNXRUNTIME,
                    common::StatusCode::INVALID_ARGUMENT,
                    "unsupported conv activation mode");
    }
    return Status::OK();
  }
  cudnnStatus_t status_ = CUDNN_STATUS_NOT_INITIALIZED;
  cudnnActivationMode_t activation_mode_;
  cudnnActivationDescriptor_t activation_desc_ = nullptr;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv<float>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime