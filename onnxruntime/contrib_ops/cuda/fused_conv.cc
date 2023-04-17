// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/providers/cuda/nn/conv.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class FusedConv : public onnxruntime::cuda::Conv<T, false> {
 public:
  using Base = onnxruntime::cuda::Conv<T, false>;
  FusedConv(const OpKernelInfo& info) : onnxruntime::cuda::Conv<T, false>(info) {
    std::string activation;
    ORT_THROW_IF_ERROR(info.GetAttr<std::string>("activation", &activation));
    ORT_THROW_IF_ERROR(MapMode(activation));
    CUDNN_CALL_THROW(cudnnCreateActivationDescriptor(&activation_desc_));
    CUDNN_CALL_THROW(cudnnSetActivationDescriptor(
        activation_desc_, activation_mode_, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        std::numeric_limits<double>::max()));
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(FusedConv);

  ~FusedConv() {
    if (activation_desc_) {
      CUDNN_CALL_THROW(cudnnDestroyActivationDescriptor(activation_desc_));
      activation_desc_ = nullptr;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    std::lock_guard<OrtMutex> lock(Base::s_.mutex);
    auto cudnnHandle = this->GetCudnnHandle(context);
    ORT_RETURN_IF_ERROR(Base::UpdateState(context, true));
    if (Base::s_.Y->Shape().Size() == 0) {
      return Status::OK();
    }
    bool has_z = nullptr != Base::s_.z_data;
    bool has_b = nullptr != Base::s_.b_data;
    typedef typename onnxruntime::cuda::ToCudaType<T>::MappedType CudaT;
    const auto alpha = onnxruntime::cuda::Consts<CudaT>::One;
    const auto beta = onnxruntime::cuda::Consts<CudaT>::Zero;
    IAllocatorUniquePtr<void> workspace = Base::GetWorkSpace(context->GetComputeStream());
    auto cudnn_status = cudnnConvolutionBiasActivationForward(cudnnHandle,
                                                              &alpha,
                                                              Base::s_.x_tensor,
                                                              Base::s_.x_data,
                                                              Base::s_.w_desc,
                                                              Base::s_.w_data,
                                                              Base::s_.conv_desc,
                                                              Base::s_.algo,
                                                              workspace.get(),
                                                              Base::s_.workspace_bytes,
                                                              has_z ? &alpha : &beta,
                                                              has_z ? Base::s_.z_tensor : Base::s_.y_tensor,
                                                              has_z ? Base::s_.z_data : Base::s_.y_data,
                                                              Base::s_.b_tensor,
                                                              has_b ? Base::s_.b_data : Base::s_.b_zero,
                                                              activation_desc_,
                                                              Base::s_.y_tensor,
                                                              Base::s_.y_data);
    if (CUDNN_STATUS_SUCCESS != cudnn_status) {
      CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(cudnnHandle,
                                                    &alpha,
                                                    Base::s_.x_tensor,
                                                    Base::s_.x_data,
                                                    Base::s_.w_desc,
                                                    Base::s_.w_data,
                                                    Base::s_.conv_desc,
                                                    Base::s_.algo,
                                                    workspace.get(),
                                                    Base::s_.workspace_bytes,
                                                    &beta,
                                                    Base::s_.y_tensor,
                                                    Base::s_.y_data));
      if (has_b) {
        CUDNN_RETURN_IF_ERROR(cudnnAddTensor(cudnnHandle, &alpha, Base::s_.b_tensor, Base::s_.b_data,
                                             &alpha, Base::s_.y_tensor, Base::s_.y_data));
      }
      if (has_z) {
        CUDNN_RETURN_IF_ERROR(cudnnAddTensor(cudnnHandle, &alpha, Base::s_.z_tensor, Base::s_.z_data,
                                             &alpha, Base::s_.y_tensor, Base::s_.y_data));
      }
      CUDNN_RETURN_IF_ERROR(cudnnActivationForward(cudnnHandle, activation_desc_, &alpha, Base::s_.y_tensor,
                                                   Base::s_.y_data, &beta, Base::s_.y_tensor, Base::s_.y_data));
    }
    if (Base::s_.post_slicing_required) {
      ORT_RETURN_IF_ERROR(onnxruntime::cuda::SliceOutUnwantedOutputSection(
          this->Stream(context), Base::s_.y_data, Base::s_.y_dims_with_adjusted_pads, Base::s_.Y->MutableDataRaw(),
          Base::s_.y_dims.GetDims(), Base::s_.slice_starts, Base::s_.slice_ends, Base::s_.slice_axes, Base::s_.element_size));
    }
    return Status::OK();
  }

 private:
  Status MapMode(const std::string& activaton_mode) {
    if (activaton_mode == "Relu") {
      activation_mode_ = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU;
    } else {
      return ORT_MAKE_STATUS(
          StatusCategory::ONNXRUNTIME, StatusCode::INVALID_ARGUMENT,
          "unsupported conv activation mode \"", activaton_mode, "\"");
    }
    return Status::OK();
  }
  cudnnActivationMode_t activation_mode_;
  cudnnActivationDescriptor_t activation_desc_ = nullptr;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv<float>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
